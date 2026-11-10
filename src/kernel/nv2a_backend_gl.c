/* OpenGL rasterizer for the decoded NV2A pushbuffer stream. The CPU executor
 * still decodes Xbox commands and transforms vertices; triangle setup,
 * fragment shading, depth, blending and pixel writes are done by the host GPU.
 * Mesa selects the installed AMD, Intel or NVIDIA driver through GLX/EGL. */
#include "nv2a_backend_gl.h"
#include "nv2a_backend.h"
#include "xbox_memory_layout.h"

#include <SDL.h>
#include <epoxy/gl.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define TEX_CACHE 128
#define TARGET_CACHE 16

typedef struct { Nv2aSurface s; GLuint fbo, color, depth; uint64_t used; } Target;
typedef struct { Nv2aTexture key; GLuint name; } CachedTexture;
typedef struct { float x,y,z,rhw; uint8_t color[4],specular[4]; float tc[4][3]; } GpuVertex;

static SDL_Window *s_window;
static SDL_GLContext s_context;
static GLuint s_program, s_vao, s_vbo;
static GLint s_size, s_alpha_enable, s_alpha_func, s_alpha_ref;
static GLint s_rc_enable,s_rc_ci,s_rc_ai,s_rc_co,s_rc_ao,s_rc_f0,s_rc_f1;
static GLint s_rc_final0,s_rc_final1,s_rc_finalc0,s_rc_finalc1,s_rc_control,s_rc_stage,s_rc_clip;
static GLint s_tex_samplers[4],s_tex_enabled[4];
static Target s_targets[TARGET_CACHE];
static uint64_t s_target_clock;
static uint64_t s_draw_calls, s_sync_calls;
static CachedTexture s_textures[TEX_CACHE];
static unsigned s_next_texture;
static GpuVertex *s_vertices;
static size_t s_vertex_capacity;
static uint8_t *s_readback;
static size_t s_readback_size;

/* The pushbuffer scanner owns the GL work thread, not boot_and_run(). SDL
 * contexts are thread-affine, so bind the context when a renderer callback is
 * entered instead of assuming startup's thread will also issue the draws. */
static int make_current(void)
{
    if (!s_context || !s_window)
        return 0;
    if (SDL_GL_GetCurrentContext() == s_context)
        return 1;
    if (SDL_GL_MakeCurrent(s_window, s_context) != 0) {
        fprintf(stderr, "[NV2A-GL] cannot activate OpenGL on render thread: %s\n",
                SDL_GetError());
        return 0;
    }
    return 1;
}

static const char *vs_src =
    "#version 330 core\n"
    "layout(location=0) in vec4 a_pos; layout(location=1) in vec4 a_color;"
    "layout(location=2) in vec4 a_specular; layout(location=3) in vec3 a_tc0;"
    "layout(location=4) in vec3 a_tc1; layout(location=5) in vec3 a_tc2;"
    "layout(location=6) in vec3 a_tc3; uniform vec2 u_size;"
    "out vec4 v_color; out vec4 v_specular; out vec3 v_tc0; out vec3 v_tc1;"
    "out vec3 v_tc2; out vec3 v_tc3;"
    "void main(){ vec2 p=a_pos.xy/u_size*2.0-1.0;"
    " float w=a_pos.w>0.000001?1.0/a_pos.w:1.0;"
    " gl_Position=vec4(p*w,(a_pos.z*2.0-1.0)*w,w);"
    " v_color=a_color; v_specular=a_specular; v_tc0=a_tc0; v_tc1=a_tc1;"
    " v_tc2=a_tc2; v_tc3=a_tc3; }";
static const char *fs_src =
    "#version 330 core\n"
    "in vec4 v_color; in vec4 v_specular; in vec3 v_tc0; in vec3 v_tc1;"
    "in vec3 v_tc2; in vec3 v_tc3; out vec4 o_color;"
    "uniform sampler2D u_tex[4]; uniform int u_tex_enabled[4];"
    "uniform int u_alpha_enable; uniform int u_alpha_func; uniform float u_alpha_ref;"
    "uniform int u_rc_enable; uniform uint u_ci[8],u_ai[8],u_co[8],u_ao[8],u_f0[8],u_f1[8];"
    "uniform uint u_final0,u_final1,u_finalc0,u_finalc1,u_control,u_stage,u_clip;"
    "float mi(float x,uint b){float p=max(x,0.0);switch(int((b>>5u)&7u)){"
    "case 0:return p;case 1:return 1.0-clamp(x,0.0,1.0);case 2:return 2.0*p-1.0;"
    "case 3:return -2.0*p+1.0;case 4:return p-0.5;case 5:return -p+0.5;"
    "case 6:return x;default:return -x;}}"
    "float mo(float x,uint m){if(m==8u)return x-0.5;if(m==16u)return x*2.0;"
    "if(m==24u)return (x-0.5)*2.0;if(m==32u)return x*4.0;if(m==48u)return x*0.5;return x;}"
    "vec4 unpackc(uint c){return vec4(float((c>>16u)&255u),float((c>>8u)&255u),float(c&255u),float(c>>24u))/255.0;}"
    "vec4 sample_stage(int i,vec2 uv){if(i==0)return texture(u_tex[0],uv);"
    "if(i==1)return texture(u_tex[1],uv);if(i==2)return texture(u_tex[2],uv);"
    "return texture(u_tex[3],uv); }"
    "vec3 rr(uint b,vec4 R[16]){vec4 r=R[int(b&15u)];vec3 x=((b&16u)!=0u)?vec3(r.a):r.rgb;"
    "return vec3(mi(x.r,b),mi(x.g,b),mi(x.b,b));}"
    "float ra(uint b,vec4 R[16]){vec4 r=R[int(b&15u)];return mi(((b&16u)!=0u)?r.a:r.b,b);}"
    "vec4 comb(vec4 v0,vec4 v1,vec4 T[4]){vec4 R[16];for(int i=0;i<16;i++)R[i]=vec4(0);"
    "R[3]=vec4(0,0,0,1);R[4]=v0;R[5]=v1;R[8]=T[0];R[9]=T[1];R[10]=T[2];R[11]=T[3];"
    "R[12].a=((u_stage&31u)!=0u)?T[0].a:1.0;uint n=min(u_control&255u,8u),flg=u_control>>8u;"
    "for(uint s=0u;s<n;s++){uint ci=u_ci[s],ai=u_ai[s],co=u_co[s],ao=u_ao[s],f=co>>12u,af=ao>>12u;"
    "uint ix=((flg&16u)!=0u)?s:0u,iy=((flg&256u)!=0u)?s:0u;R[1]=unpackc(u_f0[ix]);R[2]=unpackc(u_f1[iy]);"
    "bool mux=((flg&1u)!=0u)?R[12].a>=0.5:(int(R[12].a*255.0)&1)!=0;"
    "vec3 A=rr(ci>>24u,R),B=rr(ci>>16u,R),C=rr(ci>>8u,R),D=rr(ci,R);"
    "float aA=ra(ai>>24u,R),aB=ra(ai>>16u,R),aC=ra(ai>>8u,R),aD=ra(ai,R);"
    "float pab=(f&2u)!=0u?dot(A,B):0.0,pcd=(f&1u)!=0u?dot(C,D):0.0;"
    "vec3 p1=(f&2u)!=0u?vec3(pab):A*B, p2=(f&1u)!=0u?vec3(pcd):C*D;"
    "vec3 sm=(f&4u)!=0u?(mux?p2:p1):p1+p2;"
    "vec3 ab=clamp(vec3(mo(p1.r,f&56u),mo(p1.g,f&56u),mo(p1.b,f&56u)),-1.0,1.0);"
    "vec3 cd=clamp(vec3(mo(p2.r,f&56u),mo(p2.g,f&56u),mo(p2.b,f&56u)),-1.0,1.0);"
    "vec3 su=clamp(vec3(mo(sm.r,f&56u),mo(sm.g,f&56u),mo(sm.b,f&56u)),-1.0,1.0);"
    "float aa=clamp(mo(aA*aB,af&56u),-1.0,1.0),ac=clamp(mo(aC*aD,af&56u),-1.0,1.0);"
    "float asu=clamp(mo((af&4u)!=0u?(mux?aC*aD:aA*aB):aA*aB+aC*aD,af&56u),-1.0,1.0);"
    "int dab=int((co>>4u)&15u),dcd=int(co&15u),dsu=int((co>>8u)&15u);"
    "if(dab!=0){R[dab].rgb=ab;if((f&128u)!=0u)R[dab].a=ab.b;}"
    "if(dcd!=0){R[dcd].rgb=cd;if((f&64u)!=0u)R[dcd].a=cd.b;}if(dsu!=0)R[dsu].rgb=su;"
    "int daa=int((ao>>4u)&15u),dac=int(ao&15u),das=int((ao>>8u)&15u);"
    "if(daa!=0)R[daa].a=aa;if(dac!=0)R[dac].a=ac;if(das!=0)R[das].a=asu;R[0]=vec4(0); }"
    "vec4 outc; if(u_final0!=0u||u_final1!=0u){uint ff=u_final1&255u;R[1]=unpackc(u_finalc0);R[2]=unpackc(u_finalc1);"
    "vec3 va=((ff&64u)!=0u)?1.0-v1.rgb:v1.rgb,vb=((ff&32u)!=0u)?1.0-R[12].rgb:R[12].rgb;"
    "R[14].rgb=((ff&128u)!=0u)?clamp(va+vb,0.0,1.0):va+vb;R[14].a=0;"
    "vec3 E=rr(u_final1>>24u,R),F=rr(u_final1>>16u,R);R[15]=vec4(E*F,0);"
    "vec3 A=rr(u_final0>>24u,R),B=rr(u_final0>>16u,R),C=rr(u_final0>>8u,R),D=rr(u_final0,R);"
    "outc=vec4(D+C*(1.0-A)+B*A,ra(u_final1>>8u,R));}else outc=R[12];return clamp(outc,0.0,1.0); }"
    "void main(){vec3 tc[4]=vec3[4](v_tc0,v_tc1,v_tc2,v_tc3);vec4 T[4];"
    "for(int i=0;i<4;i++){uint mode=(u_stage>>(uint(i)*5u))&31u;"
    "if(mode==0u)T[i]=vec4(0,0,0,1);else if(mode==4u)T[i]=clamp(vec4(tc[i].xy,0,1),0,1);"
    "else if(u_tex_enabled[i]!=0)T[i]=sample_stage(i,tc[i].xy/(abs(tc[i].z)>0.000001?tc[i].z:1.0));"
    "else T[i]=vec4(1);if(mode==5u){T[i]=vec4(0);for(int j=0;j<4;j++){bool ge=((u_clip>>(uint(i)*4u+uint(j)))&1u)!=0u;float c=j<2?tc[i][j]:0.0;if(ge?c<0.0:c>=0.0)discard;}}}"
    "vec4 c=v_color;if(u_rc_enable!=0)c=comb(v_color,v_specular,T);else if(u_tex_enabled[0]!=0)c*=T[0];"
    " if(u_alpha_enable!=0){bool pass=false;"
    " if(u_alpha_func==512)pass=false; else if(u_alpha_func==513)pass=c.a<u_alpha_ref;"
    " else if(u_alpha_func==514)pass=c.a==u_alpha_ref; else if(u_alpha_func==515)pass=c.a<=u_alpha_ref;"
    " else if(u_alpha_func==516)pass=c.a>u_alpha_ref; else if(u_alpha_func==517)pass=c.a!=u_alpha_ref;"
    " else if(u_alpha_func==518)pass=c.a>=u_alpha_ref; else pass=true; if(!pass)discard;}"
    " o_color=c; }";

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint sh=glCreateShader(type); GLint ok=0; char log[1024];
    glShaderSource(sh,1,&src,NULL); glCompileShader(sh); glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ glGetShaderInfoLog(sh,sizeof log,NULL,log); fprintf(stderr,"[NV2A-GL] shader: %s\n",log); glDeleteShader(sh); return 0; }
    return sh;
}

static Target *target_for_va(uint32_t va)
{
    unsigned i;
    for(i=0;i<TARGET_CACHE;i++)
        if(s_targets[i].fbo && s_targets[i].s.color_va==va)
            return &s_targets[i];
    return NULL;
}

static void destroy_target(Target *t)
{
    if(t->fbo)glDeleteFramebuffers(1,&t->fbo);
    if(t->color)glDeleteTextures(1,&t->color);
    if(t->depth)glDeleteRenderbuffers(1,&t->depth);
    memset(t,0,sizeof *t);
}

static Target *ensure_target(const Nv2aSurface *s)
{
    unsigned i,slot=0;uint64_t oldest=UINT64_MAX;Target *t=target_for_va(s->color_va);
    if(t && t->s.width==s->width && t->s.height==s->height && t->s.pitch==s->pitch){
        t->used=++s_target_clock;return t;
    }
    if(t)destroy_target(t);
    for(i=0;i<TARGET_CACHE;i++){
        if(!s_targets[i].fbo){slot=i;oldest=0;break;}
        if(s_targets[i].used<oldest){oldest=s_targets[i].used;slot=i;}
    }
    t=&s_targets[slot];destroy_target(t);t->s=*s;t->used=++s_target_clock;
    glGenTextures(1,&t->color); glBindTexture(GL_TEXTURE_2D,t->color);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,(GLsizei)s->width,(GLsizei)s->height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glGenFramebuffers(1,&t->fbo); glBindFramebuffer(GL_FRAMEBUFFER,t->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->color,0);
    glGenRenderbuffers(1,&t->depth); glBindRenderbuffer(GL_RENDERBUFFER,t->depth);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,(GLsizei)s->width,(GLsizei)s->height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_RENDERBUFFER,t->depth);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){ fprintf(stderr,"[NV2A-GL] framebuffer setup failed for 0x%08X\n",s->color_va); destroy_target(t);return NULL; }
    return t;
}

static int same_tex(const Nv2aTexture *a,const Nv2aTexture *b)
{ return a->offset==b->offset && a->width==b->width && a->height==b->height
      && a->color==b->color && a->pitch==b->pitch && a->levels==b->levels
      && a->filter==b->filter && a->addr_u==b->addr_u && a->addr_v==b->addr_v; }

/* XMV and other streaming surfaces can rewrite a linear full-frame texture at
 * the same guest VA every frame. A normal immutable-texture cache silently
 * freezes the first frame (often the black frame before the decoder starts).
 * Update this one common framebuffer format directly from guest memory; the
 * pixels stay GPU-rendered and avoid a per-pixel CPU decode. */
static void refresh_streaming_frame(const Nv2aTexture *t, GLuint name)
{
    GLint old_row_length;
    const void *pixels;
    if(t->color!=0x12 || t->width!=640 || t->height!=480
       || t->pitch<t->width*4 || (t->pitch&3))return;
    pixels=(const void*)((uintptr_t)xbox_GetMemoryOffset()+t->offset);
    glBindTexture(GL_TEXTURE_2D,name);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH,&old_row_length);
    glPixelStorei(GL_UNPACK_ALIGNMENT,4);
    glPixelStorei(GL_UNPACK_ROW_LENGTH,(GLint)(t->pitch/4));
    glTexSubImage2D(GL_TEXTURE_2D,0,0,0,(GLsizei)t->width,(GLsizei)t->height,
                    GL_BGRA,GL_UNSIGNED_BYTE,pixels);
    glPixelStorei(GL_UNPACK_ROW_LENGTH,old_row_length);
}

static GLuint texture_for(const Nv2aTexture *t)
{
    uint32_t i; GLuint n; uint32_t *px;
    if(!t || !t->width || !t->height || (uint64_t)t->width*t->height>16777216u) return 0;
    {
        Target *rt=target_for_va(t->offset);
        if(rt){rt->used=++s_target_clock;glBindTexture(GL_TEXTURE_2D,rt->color);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,(t->filter&0x00ff0000u)?GL_LINEAR:GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,t->addr_u==3?GL_CLAMP_TO_EDGE:GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,t->addr_v==3?GL_CLAMP_TO_EDGE:GL_REPEAT);
            return rt->color;}
    }
    for(i=0;i<TEX_CACHE;i++) if(s_textures[i].name && same_tex(&s_textures[i].key,t)){
        refresh_streaming_frame(t,s_textures[i].name);
        return s_textures[i].name;
    }
    px=(uint32_t*)malloc((size_t)t->width*t->height*sizeof *px); if(!px)return 0;
    if(!nv2a_backend_decode_texture(t,px)){free(px);return 0;}
    glGenTextures(1,&n); glBindTexture(GL_TEXTURE_2D,n);
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,(GLsizei)t->width,(GLsizei)t->height,0,GL_BGRA,GL_UNSIGNED_BYTE,px);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,(t->filter&0x00ff0000u)?GL_LINEAR:GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,t->addr_u==3?GL_CLAMP_TO_EDGE:GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,t->addr_v==3?GL_CLAMP_TO_EDGE:GL_REPEAT);
    free(px);
    i=s_next_texture++%TEX_CACHE; if(s_textures[i].name)glDeleteTextures(1,&s_textures[i].name);
    s_textures[i].key=*t; s_textures[i].name=n; return n;
}

static void backend_clear(const Nv2aSurface *s,const Nv2aRenderState *rs,uint32_t flags,uint32_t argb,uint32_t zs)
{
    Target *t;
    (void)rs; if(!make_current() || !(t=ensure_target(s)))return; glBindFramebuffer(GL_FRAMEBUFFER,t->fbo); glViewport(0,0,(GLsizei)s->width,(GLsizei)s->height);
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDepthMask(GL_TRUE);glStencilMask(0xff);glDisable(GL_SCISSOR_TEST);
    GLbitfield bits=0;
    if(flags&0xf0){glClearColor(((argb>>16)&255)/255.f,((argb>>8)&255)/255.f,(argb&255)/255.f,((argb>>24)&255)/255.f);bits|=GL_COLOR_BUFFER_BIT;}
    if(flags&1){glClearDepth(((zs>>8)&0xffffffu)/16777215.0);bits|=GL_DEPTH_BUFFER_BIT;}
    if(flags&2){glClearStencil(zs&255);bits|=GL_STENCIL_BUFFER_BIT;}
    glClear(bits);
}

static void backend_draw(const Nv2aSurface *s,const Nv2aBatch *b)
{
    uint32_t i; GpuVertex *v; GLuint tex[4]={0}; const Nv2aRenderState *r=b->state;Target *t;
    if(!b->count || !make_current() || !(t=ensure_target(s)))return;
    if((size_t)b->count>s_vertex_capacity){
        size_t cap=s_vertex_capacity?s_vertex_capacity:256;
        GpuVertex *p;
        while(cap<(size_t)b->count)cap*=2;
        p=(GpuVertex*)realloc(s_vertices,cap*sizeof *p);if(!p)return;
        s_vertices=p;s_vertex_capacity=cap;
    }
    v=s_vertices;
    for(i=0;i<b->count;i++){const Nv2aVertex *a=&b->vertices[i]; v[i].x=a->x;v[i].y=a->y;v[i].z=a->z;v[i].rhw=a->rhw;
        v[i].color[0]=(uint8_t)(a->diffuse>>16);v[i].color[1]=(uint8_t)(a->diffuse>>8);v[i].color[2]=(uint8_t)a->diffuse;v[i].color[3]=(uint8_t)(a->diffuse>>24);
        v[i].specular[0]=(uint8_t)(a->specular>>16);v[i].specular[1]=(uint8_t)(a->specular>>8);v[i].specular[2]=(uint8_t)a->specular;v[i].specular[3]=(uint8_t)(a->specular>>24);
        {const float uv[4][2]={{a->u,a->v},{a->u1,a->v1},{a->u2,a->v2},{a->u3,a->v3}};const float q[4]={a->q0,a->q1,a->q2,a->q3};int st;
        for(st=0;st<4;st++){const Nv2aTexture *tt=b->textures[st];float w=tt&&tt->width?(float)tt->width:1.f,h=tt&&tt->height?(float)tt->height:1.f;v[i].tc[st][0]=uv[st][0]/w;v[i].tc[st][1]=uv[st][1]/h;v[i].tc[st][2]=q[st];}}
    }
    for(i=0;i<4;i++)if(b->textures[i])tex[i]=texture_for(b->textures[i]);
    if(s_draw_calls<16){
        fprintf(stderr,"[NV2A-GL] draw %llu surface=0x%08X verts=%u first=(%.2f,%.2f,%.4f) rgba=%08X spec=%08X uv0=(%.4f,%.4f,%.4f) rc=%u control=%08X stage=%08X icw=%08X/%08X ocw=%08X/%08X final=%08X/%08X alpha=%u:%08X mask=%08X\n",
                (unsigned long long)s_draw_calls,s->color_va,b->count,
                b->vertices[0].x,b->vertices[0].y,b->vertices[0].z,
                b->vertices[0].diffuse,b->vertices[0].specular,
                v[0].tc[0][0],v[0].tc[0][1],v[0].tc[0][2],r->combiner_enable,
                r->combiner.control,r->combiner.stage_program,
                r->combiner.color_icw[0],r->combiner.alpha_icw[0],
                r->combiner.color_ocw[0],r->combiner.alpha_ocw[0],
                r->combiner.final0,r->combiner.final1,
                r->alpha_test_enable,r->alpha_func,r->color_mask);
        for(i=0;i<4;i++)if(b->textures[i])
            fprintf(stderr,"[NV2A-GL]   tex%u offset=%08X size=%ux%u fmt=%02X enabled=%u\n",
                    i,b->textures[i]->offset,b->textures[i]->width,b->textures[i]->height,
                    b->textures[i]->color,tex[i]!=0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER,t->fbo);glViewport(0,0,(GLsizei)s->width,(GLsizei)s->height);glUseProgram(s_program);glBindVertexArray(s_vao);
    for(i=0;i<7;i++)glEnableVertexAttribArray(i);
    glUniform2f(s_size,(float)s->width,(float)s->height);
    glUniform1i(s_alpha_enable,r->alpha_test_enable?1:0);glUniform1i(s_alpha_func,(GLint)r->alpha_func);glUniform1f(s_alpha_ref,r->alpha_ref/255.f);
    for(i=0;i<4;i++){glActiveTexture(GL_TEXTURE0+i);glBindTexture(GL_TEXTURE_2D,tex[i]);glUniform1i(s_tex_samplers[i],(GLint)i);glUniform1i(s_tex_enabled[i],tex[i]?1:0);}
    glUniform1i(s_rc_enable,r->combiner_enable?1:0);
    {const Nv2aCombiner *c=&r->combiner;glUniform1uiv(s_rc_ci,8,c->color_icw);glUniform1uiv(s_rc_ai,8,c->alpha_icw);glUniform1uiv(s_rc_co,8,c->color_ocw);glUniform1uiv(s_rc_ao,8,c->alpha_ocw);glUniform1uiv(s_rc_f0,8,c->factor0);glUniform1uiv(s_rc_f1,8,c->factor1);
     glUniform1ui(s_rc_final0,c->final0);glUniform1ui(s_rc_final1,c->final1);glUniform1ui(s_rc_finalc0,c->final_c0);glUniform1ui(s_rc_finalc1,c->final_c1);glUniform1ui(s_rc_control,c->control);glUniform1ui(s_rc_stage,c->stage_program);glUniform1ui(s_rc_clip,r->clip_plane_mode);}
    if(r->blend_enable){glEnable(GL_BLEND);glBlendFunc((GLenum)r->blend_src,(GLenum)r->blend_dst);glBlendEquation((GLenum)r->blend_eq);}else glDisable(GL_BLEND);
    if(r->depth_test_enable){glEnable(GL_DEPTH_TEST);glDepthFunc((GLenum)r->depth_func);}else glDisable(GL_DEPTH_TEST);glDepthMask(r->depth_write?GL_TRUE:GL_FALSE);
    glColorMask((r->color_mask&0x00010000u)!=0,(r->color_mask&0x100u)!=0,(r->color_mask&1)!=0,(r->color_mask&0x01000000u)!=0);
    if(r->cull_enable){glEnable(GL_CULL_FACE);glCullFace(r->cull_face==0x0404?GL_FRONT:GL_BACK);glFrontFace(r->front_face==0x0900?GL_CW:GL_CCW);}else glDisable(GL_CULL_FACE);
    if(r->stencil_enable){glEnable(GL_STENCIL_TEST);glStencilFunc((GLenum)r->stencil_func,(GLint)r->stencil_ref,r->stencil_func_mask);glStencilMask(r->stencil_write_mask);glStencilOp((GLenum)r->stencil_fail,(GLenum)r->stencil_zfail,(GLenum)r->stencil_zpass);}else glDisable(GL_STENCIL_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,s_vbo);glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)((size_t)b->count*sizeof *v),v,GL_STREAM_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof *v,(void*)offsetof(GpuVertex,x));
    glVertexAttribPointer(1,4,GL_UNSIGNED_BYTE,GL_TRUE,sizeof *v,(void*)offsetof(GpuVertex,color));
    glVertexAttribPointer(2,4,GL_UNSIGNED_BYTE,GL_TRUE,sizeof *v,(void*)offsetof(GpuVertex,specular));
    for(i=0;i<4;i++)glVertexAttribPointer(3+i,3,GL_FLOAT,GL_FALSE,sizeof *v,(void*)offsetof(GpuVertex,tc)+i*sizeof v[0].tc[0]);
    glDrawArrays(GL_TRIANGLES,0,(GLsizei)b->count);
    if(s_draw_calls<16){GLenum e=glGetError();if(e!=GL_NO_ERROR)fprintf(stderr,"[NV2A-GL] draw GL error 0x%04X\n",e);}
    s_draw_calls++;
}

static void backend_flip(void) { if(make_current())glFlush(); }

static void backend_sync(uint32_t surface_va)
{
    size_t need; uint32_t y; uint8_t *dst; Target *t;const Nv2aSurface *s;uint64_t seq=++s_sync_calls;
    if(!make_current())return;
    if(!(t=target_for_va(surface_va))){
        if(seq<=12){unsigned i;fprintf(stderr,"[NV2A-GL] sync %llu surface=0x%08X missing; cached:",
                (unsigned long long)seq,surface_va);for(i=0;i<TARGET_CACHE;i++)if(s_targets[i].fbo)fprintf(stderr," 0x%08X",s_targets[i].s.color_va);fputc('\n',stderr);}
        return;
    }
    s=&t->s;if(!s->color_va || !s->width || !s->height)return;
    glBindFramebuffer(GL_FRAMEBUFFER,t->fbo);glFinish();need=(size_t)s->width*s->height*4;
    if(need>s_readback_size){uint8_t *p=(uint8_t*)realloc(s_readback,need);if(!p)return;s_readback=p;s_readback_size=need;}
    glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,(GLsizei)s->width,(GLsizei)s->height,GL_BGRA,GL_UNSIGNED_BYTE,s_readback);
    if(seq<=8 || seq%120==0){
        size_t p,nz=0;GLenum e=glGetError();
        for(p=0;p<need;p+=4)if(s_readback[p]||s_readback[p+1]||s_readback[p+2])nz++;
        fprintf(stderr,"[NV2A-GL] sync %llu surface=0x%08X %ux%u nonblack=%llu/%llu GL=0x%04X\n",
                (unsigned long long)seq,surface_va,s->width,s->height,
                (unsigned long long)nz,(unsigned long long)((size_t)s->width*s->height),e);
    }
    dst=(uint8_t*)((uintptr_t)xbox_GetMemoryOffset()+s->color_va);
    if(s->bytes_per_pixel==4){for(y=0;y<s->height;y++)memcpy(dst+(size_t)y*s->pitch,s_readback+(size_t)y*s->width*4,(size_t)s->width*4);}
    else if(s->bytes_per_pixel==2){for(y=0;y<s->height;y++){uint32_t x;uint16_t *d=(uint16_t*)(dst+(size_t)y*s->pitch);const uint8_t *p=s_readback+(size_t)y*s->width*4;for(x=0;x<s->width;x++)d[x]=(uint16_t)(((p[x*4+2]>>3)<<11)|((p[x*4+1]>>2)<<5)|(p[x*4]>>3));}}
}

int nv2a_gl_backend_init(void)
{
    static Nv2aBackend backend={backend_clear,backend_draw,backend_flip};GLuint vs,fs;GLint ok=0;
    if(!(SDL_WasInit(SDL_INIT_VIDEO)&SDL_INIT_VIDEO) && SDL_InitSubSystem(SDL_INIT_VIDEO)!=0){fprintf(stderr,"[NV2A-GL] SDL video init failed: %s\n",SDL_GetError());return 0;}
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,0);
    s_window=SDL_CreateWindow("UC2 NV2A OpenGL",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,16,16,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
    if(!s_window){fprintf(stderr,"[NV2A-GL] GPU window creation failed: %s\n",SDL_GetError());return 0;}
    s_context=SDL_GL_CreateContext(s_window);if(!s_context){fprintf(stderr,"[NV2A-GL] GPU context creation failed: %s\n",SDL_GetError());SDL_DestroyWindow(s_window);s_window=NULL;return 0;}
    if(SDL_GL_MakeCurrent(s_window,s_context)!=0){fprintf(stderr,"[NV2A-GL] GPU context activation failed: %s\n",SDL_GetError());return 0;}
    fprintf(stderr,"[NV2A-GL] hardware OpenGL %s | %s\n",glGetString(GL_VERSION),glGetString(GL_RENDERER));
    if(!glGetString(GL_RENDERER)||strstr((const char*)glGetString(GL_RENDERER),"llvmpipe")||strstr((const char*)glGetString(GL_RENDERER),"softpipe")){fprintf(stderr,"[NV2A-GL] software OpenGL renderer rejected; install/enable a hardware GPU driver\n");return 0;}
    vs=compile_shader(GL_VERTEX_SHADER,vs_src);fs=compile_shader(GL_FRAGMENT_SHADER,fs_src);if(!vs||!fs)return 0;
    s_program=glCreateProgram();glAttachShader(s_program,vs);glAttachShader(s_program,fs);glLinkProgram(s_program);glGetProgramiv(s_program,GL_LINK_STATUS,&ok);glDeleteShader(vs);glDeleteShader(fs);if(!ok){fprintf(stderr,"[NV2A-GL] shader link failed\n");return 0;}
    s_size=glGetUniformLocation(s_program,"u_size");s_alpha_enable=glGetUniformLocation(s_program,"u_alpha_enable");s_alpha_func=glGetUniformLocation(s_program,"u_alpha_func");s_alpha_ref=glGetUniformLocation(s_program,"u_alpha_ref");
    s_rc_enable=glGetUniformLocation(s_program,"u_rc_enable");s_rc_ci=glGetUniformLocation(s_program,"u_ci[0]");s_rc_ai=glGetUniformLocation(s_program,"u_ai[0]");s_rc_co=glGetUniformLocation(s_program,"u_co[0]");s_rc_ao=glGetUniformLocation(s_program,"u_ao[0]");s_rc_f0=glGetUniformLocation(s_program,"u_f0[0]");s_rc_f1=glGetUniformLocation(s_program,"u_f1[0]");
    s_rc_final0=glGetUniformLocation(s_program,"u_final0");s_rc_final1=glGetUniformLocation(s_program,"u_final1");s_rc_finalc0=glGetUniformLocation(s_program,"u_finalc0");s_rc_finalc1=glGetUniformLocation(s_program,"u_finalc1");s_rc_control=glGetUniformLocation(s_program,"u_control");s_rc_stage=glGetUniformLocation(s_program,"u_stage");s_rc_clip=glGetUniformLocation(s_program,"u_clip");
    for(int i=0;i<4;i++){char name[32];snprintf(name,sizeof name,"u_tex[%d]",i);s_tex_samplers[i]=glGetUniformLocation(s_program,name);snprintf(name,sizeof name,"u_tex_enabled[%d]",i);s_tex_enabled[i]=glGetUniformLocation(s_program,name);}
    glGenVertexArrays(1,&s_vao);glGenBuffers(1,&s_vbo);nv2a_backend_register_with_surface_sync(&backend,backend_sync);
    /* Leave it unowned; the executor thread will bind it on its first call. */
    SDL_GL_MakeCurrent(s_window,NULL);
    return 1;
}

void nv2a_gl_backend_shutdown(void)
{
    unsigned i;nv2a_backend_register(NULL);if(!s_context)return;
    if(SDL_GL_GetCurrentContext()!=s_context && SDL_GL_MakeCurrent(s_window,s_context)!=0)return;
    for(i=0;i<TEX_CACHE;i++)if(s_textures[i].name)glDeleteTextures(1,&s_textures[i].name);
    for(i=0;i<TARGET_CACHE;i++)destroy_target(&s_targets[i]);
    if(s_vbo)glDeleteBuffers(1,&s_vbo);
    if(s_vao)glDeleteVertexArrays(1,&s_vao);
    if(s_program)glDeleteProgram(s_program);
    free(s_readback);
    free(s_vertices);s_vertices=NULL;s_vertex_capacity=0;
    SDL_GL_DeleteContext(s_context);SDL_DestroyWindow(s_window);s_context=NULL;s_window=NULL;
}
