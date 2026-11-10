/* OpenGL rasterizer for the decoded NV2A pushbuffer stream. The CPU executor
 * still decodes Xbox commands and transforms vertices; triangle setup,
 * fragment shading, depth, blending and pixel writes are done by the host GPU.
 * Mesa selects the installed AMD, Intel or NVIDIA driver through GLX/EGL. */
#include "nv2a_backend_gl.h"
#include "nv2a_backend.h"
#include "xbox_memory_layout.h"
#include "../nv2a/nv2a_regs.h"

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
typedef struct {
    uint64_t seq;
    uint32_t surface, count, diffuse, tex0, tex1, stage, color_mask;
    uint32_t control, ci0, ai0, co0, ao0, final0, final1;
    uint32_t depth_func, depth_write, blend_src, blend_dst, clip_mode;
    uint32_t tex0_size, tex0_pitch, tex0_format;
    float min_x, max_x, min_y, max_y, min_z, max_z, min_rhw, max_rhw;
    uint8_t combiner, alpha_test, blend, depth, cull, cull_face, front_face;
} DrawTrace;

static SDL_Window *s_window;
static SDL_GLContext s_context;
static GLuint s_program, s_vao, s_vbo;
static GLuint s_pvideo_program, s_pvideo_vao, s_pvideo_vbo, s_pvideo_texture;
static GLint s_pv_base, s_pv_yuy, s_pv_window, s_pv_guest, s_pv_origin;
static GLint s_pv_out_size, s_pv_in_size, s_pv_start, s_pv_step, s_pv_key_on, s_pv_key;
static uint32_t s_pvideo_regs[0x1000 / 4];
static int s_pvideo_active;
static uint64_t s_pvideo_frames;
static unsigned s_pvideo_bank;
static GLint s_size, s_alpha_enable, s_alpha_func, s_alpha_ref;
static GLint s_rc_enable,s_rc_ci,s_rc_ai,s_rc_co,s_rc_ao,s_rc_f0,s_rc_f1;
static GLint s_rc_final0,s_rc_final1,s_rc_finalc0,s_rc_finalc1,s_rc_control,s_rc_stage,s_rc_clip;
static GLint s_tex_samplers[4],s_tex_enabled[4];
static Target s_targets[TARGET_CACHE];
static uint64_t s_target_clock;
static uint64_t s_draw_calls, s_sync_calls;
static DrawTrace s_draw_trace[8];
static CachedTexture s_textures[TEX_CACHE];
static unsigned s_next_texture;
static GpuVertex *s_vertices;
static size_t s_vertex_capacity;
static uint8_t *s_readback;
static size_t s_readback_size;
static int s_owns_window;

extern void *xbox_FramebufferWindowOpenGLWindow(void);

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

/* PVIDEO carries Xbox movie frames as YUY2 and scans them over the 3D
 * framebuffer. Keep both conversion and composition on the host GPU. */
static const char *pvideo_vs_src =
    "#version 330 core\n"
    "layout(location=0) in vec2 p; void main(){gl_Position=vec4(p,0.0,1.0);}";
static const char *pvideo_fs_src =
    "#version 330 core\n"
    "out vec4 color; uniform sampler2D u_base; uniform sampler2D u_yuy;"
    "uniform vec2 u_window,u_guest,u_origin,u_out_size,u_in_size,u_start,u_step;"
    "uniform int u_key_on; uniform vec3 u_key;"
    "void main(){vec2 q=vec2(gl_FragCoord.x*u_guest.x/u_window.x,"
    "(u_window.y-gl_FragCoord.y)*u_guest.y/u_window.y);"
    "vec4 under=texture(u_base,vec2(q.x/u_guest.x,q.y/u_guest.y));"
    "if(q.x<u_origin.x||q.y<u_origin.y||q.x>=u_origin.x+u_out_size.x||"
    "q.y>=u_origin.y+u_out_size.y)discard;"
    "if(u_key_on!=0&&any(notEqual(floor(under.rgb*255.0+0.5),u_key)))discard;"
    "vec2 st=u_start+(q-u_origin)*u_step;"
    "if(any(lessThan(st,vec2(0.0)))||st.x>=u_in_size.x||st.y>=u_in_size.y)discard;"
    "float pair=floor(st.x*0.5);"
    "vec4 yuv=texture(u_yuy,vec2((pair+0.5)/ceil(u_in_size.x*0.5),"
    "(floor(st.y)+0.5)/u_in_size.y));"
    "float y=(mod(floor(st.x),2.0)<1.0?yuv.r:yuv.b)*255.0-16.0;"
    "float u=yuv.g*255.0-128.0,v=yuv.a*255.0-128.0;"
    "vec3 rgb=clamp(vec3(298.0*y+409.0*v,298.0*y-100.0*u-208.0*v,"
    "298.0*y+516.0*u)/255.0,0.0,255.0)/255.0;color=vec4(rgb,1.0);}";

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint sh=glCreateShader(type); GLint ok=0; char log[1024];
    glShaderSource(sh,1,&src,NULL); glCompileShader(sh); glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ glGetShaderInfoLog(sh,sizeof log,NULL,log); fprintf(stderr,"[NV2A-GL] shader: %s\n",log); glDeleteShader(sh); return 0; }
    return sh;
}

/* On POSIX the 16 MB NV2A aperture is mapped RAM, so the Xbox's PVIDEO
 * register writes land at FD008000 without passing through the Windows NV2A
 * model. Read that register block here and latch each submitted YUY2 bank. */
static int pvideo_latch(void)
{
    volatile uint32_t *regs = (volatile uint32_t *)
        ((uintptr_t)xbox_GetMemoryOffset() + 0xFD008000u);
    uint32_t snapshot[0x1000 / 4], pending;
    unsigned i, bank;
    if (!xbox_GetMemoryBase()) return 0;
    for (i = 0; i < 0x1000 / 4; ++i)
        snapshot[i] = __atomic_load_n(&regs[i], __ATOMIC_RELAXED);
    if (snapshot[NV_PVIDEO_STOP / 4] & 1u) {
        s_pvideo_active = 0;
        return 0;
    }
    pending = snapshot[NV_PVIDEO_BUFFER / 4];
    for (bank = 0; bank < 2; ++bank) {
        uint32_t bit = 1u << (bank * 4u);
        uint32_t format, pitch, dimensions, width, height;
        uint64_t physical, bytes;
        const void *pixels;
        GLint old_row_length = 0;
        if (!(pending & bit)) continue;
        format = snapshot[(NV_PVIDEO_FORMAT + bank * 4u) / 4u];
        pitch = format & NV_PVIDEO_FORMAT_PITCH;
        dimensions = snapshot[(NV_PVIDEO_SIZE_IN + bank * 4u) / 4u];
        width = dimensions & NV_PVIDEO_SIZE_IN_WIDTH;
        height = (dimensions & NV_PVIDEO_SIZE_IN_HEIGHT) >> 16;
        bytes = (uint64_t)pitch * height;
        physical = (uint64_t)snapshot[(NV_PVIDEO_BASE + bank * 4u) / 4u] +
                   snapshot[(NV_PVIDEO_OFFSET + bank * 4u) / 4u];
        if (!width || !height || (width & 1u) || pitch < width * 2u ||
            (pitch & 3u) ||
            ((format & NV_PVIDEO_FORMAT_COLOR) >> 16) !=
                NV_PVIDEO_FORMAT_COLOR_LE_CR8YB8CB8YA8 ||
            bytes > XBOX_TOTAL_RAM || physical + bytes > XBOX_TOTAL_RAM)
            continue;
        pixels = (const void *)((uintptr_t)xbox_GetMemoryOffset() +
                                0x80000000u + (uintptr_t)physical);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, s_pvideo_texture);
        glGetIntegerv(GL_UNPACK_ROW_LENGTH, &old_row_length);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, (GLint)(pitch / 4u));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, (GLsizei)(width / 2u),
                     (GLsizei)height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, old_row_length);
        memcpy(s_pvideo_regs, snapshot, sizeof(s_pvideo_regs));
        s_pvideo_bank = bank;
        __atomic_fetch_and(&regs[NV_PVIDEO_BUFFER / 4], ~bit, __ATOMIC_RELEASE);
        s_pvideo_active = 1;
        ++s_pvideo_frames;
        if (s_pvideo_frames == 1 || s_pvideo_frames % 300 == 0)
            fprintf(stderr, "[NV2A-GL] PVIDEO YUY2 frame %llu: %ux%u pitch=%u bank=%u\n",
                    (unsigned long long)s_pvideo_frames, width, height, pitch, bank);
        break;
    }
    return s_pvideo_active;
}

static void pvideo_present(const Target *target, int window_w, int window_h)
{
    uint32_t bank = s_pvideo_bank;
    uint32_t format, point, size, point_in;
    if (!pvideo_latch() || !target || window_w <= 0 || window_h <= 0)
        return;
    bank = s_pvideo_bank;
    format = s_pvideo_regs[(NV_PVIDEO_FORMAT + bank * 4u) / 4u];
    point = s_pvideo_regs[(NV_PVIDEO_POINT_OUT + bank * 4u) / 4u];
    size = s_pvideo_regs[(NV_PVIDEO_SIZE_OUT + bank * 4u) / 4u];
    point_in = s_pvideo_regs[(NV_PVIDEO_POINT_IN + bank * 4u) / 4u];
    glViewport(0, 0, window_w, window_h);
    glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST); glDisable(GL_CULL_FACE);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glUseProgram(s_pvideo_program); glBindVertexArray(s_pvideo_vao);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, target->color);
    glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, s_pvideo_texture);
    glUniform1i(s_pv_base, 0); glUniform1i(s_pv_yuy, 1);
    glUniform2f(s_pv_window, (float)window_w, (float)window_h);
    glUniform2f(s_pv_guest, (float)target->s.width, (float)target->s.height);
    glUniform2f(s_pv_origin, (float)(point & NV_PVIDEO_POINT_OUT_X),
                (float)((point & NV_PVIDEO_POINT_OUT_Y) >> 16));
    glUniform2f(s_pv_out_size, (float)(size & NV_PVIDEO_SIZE_OUT_WIDTH),
                (float)((size & NV_PVIDEO_SIZE_OUT_HEIGHT) >> 16));
    glUniform2f(s_pv_in_size,
                (float)(s_pvideo_regs[(NV_PVIDEO_SIZE_IN + bank * 4u) / 4u] &
                        NV_PVIDEO_SIZE_IN_WIDTH),
                (float)((s_pvideo_regs[(NV_PVIDEO_SIZE_IN + bank * 4u) / 4u] &
                         NV_PVIDEO_SIZE_IN_HEIGHT) >> 16));
    glUniform2f(s_pv_start,
                (float)((point_in & NV_PVIDEO_POINT_IN_S) / 16.0),
                (float)(((point_in >> 17) << 17) / 1048576.0));
    glUniform2f(s_pv_step,
                (float)(s_pvideo_regs[(NV_PVIDEO_DS_DX + bank * 4u) / 4u] / 1048576.0),
                (float)(s_pvideo_regs[(NV_PVIDEO_DT_DY + bank * 4u) / 4u] / 1048576.0));
    glUniform1i(s_pv_key_on,
                ((format & NV_PVIDEO_FORMAT_DISPLAY) >> 20) ==
                    NV_PVIDEO_FORMAT_DISPLAY_COLOR_KEY);
    glUniform3f(s_pv_key,
                (float)((s_pvideo_regs[NV_PVIDEO_COLOR_KEY / 4] >> 16) & 255u),
                (float)((s_pvideo_regs[NV_PVIDEO_COLOR_KEY / 4] >> 8) & 255u),
                (float)(s_pvideo_regs[NV_PVIDEO_COLOR_KEY / 4] & 255u));
    glDrawArrays(GL_TRIANGLES, 0, 6);
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
    float min_x=0.0f,max_x=0.0f,min_y=0.0f,max_y=0.0f;
    float min_z=0.0f,max_z=0.0f,min_rhw=0.0f,max_rhw=0.0f;
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
        if(i==0){min_x=max_x=a->x;min_y=max_y=a->y;min_z=max_z=a->z;min_rhw=max_rhw=a->rhw;}
        else{if(a->x<min_x)min_x=a->x;if(a->x>max_x)max_x=a->x;if(a->y<min_y)min_y=a->y;if(a->y>max_y)max_y=a->y;if(a->z<min_z)min_z=a->z;if(a->z>max_z)max_z=a->z;if(a->rhw<min_rhw)min_rhw=a->rhw;if(a->rhw>max_rhw)max_rhw=a->rhw;}
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
    {
        DrawTrace *d=&s_draw_trace[s_draw_calls%8];
        d->seq=s_draw_calls; d->surface=s->color_va; d->count=b->count;
        d->diffuse=b->vertices[0].diffuse;
        d->tex0=b->textures[0]?b->textures[0]->offset:0;
        d->tex1=b->textures[1]?b->textures[1]->offset:0;
        d->stage=r->combiner.stage_program; d->color_mask=r->color_mask;
        d->control=r->combiner.control;d->ci0=r->combiner.color_icw[0];d->ai0=r->combiner.alpha_icw[0];
        d->co0=r->combiner.color_ocw[0];d->ao0=r->combiner.alpha_ocw[0];
        d->final0=r->combiner.final0;d->final1=r->combiner.final1;
        d->depth_func=r->depth_func;d->depth_write=r->depth_write;
        d->blend_src=r->blend_src;d->blend_dst=r->blend_dst;d->clip_mode=r->clip_plane_mode;
        d->tex0_size=b->textures[0]?((b->textures[0]->width&0xffffu)<<16)|(b->textures[0]->height&0xffffu):0;
        d->tex0_pitch=b->textures[0]?b->textures[0]->pitch:0;
        d->tex0_format=b->textures[0]?b->textures[0]->color:0;
        d->min_x=min_x;d->max_x=max_x;d->min_y=min_y;d->max_y=max_y;
        d->min_z=min_z;d->max_z=max_z;d->min_rhw=min_rhw;d->max_rhw=max_rhw;
        d->combiner=r->combiner_enable; d->alpha_test=r->alpha_test_enable;
        d->blend=r->blend_enable; d->depth=r->depth_test_enable;
        d->cull=r->cull_enable;d->cull_face=(uint8_t)r->cull_face;d->front_face=(uint8_t)r->front_face;
    }
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
    /* glReadPixels from client memory is itself ordered after prior draws and
     * waits for this FBO's results. An explicit glFinish here forced a second
     * full-pipeline wait on every guest flip and made frame pacing spiky. */
    glBindFramebuffer(GL_FRAMEBUFFER,t->fbo);need=(size_t)s->width*s->height*4;
    if(need>s_readback_size){uint8_t *p=(uint8_t*)realloc(s_readback,need);if(!p)return;s_readback=p;s_readback_size=need;}
    glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,(GLsizei)s->width,(GLsizei)s->height,GL_BGRA,GL_UNSIGNED_BYTE,s_readback);
    if(seq<=8 || seq%120==0){
        size_t p,nz=0;GLenum e=glGetError();
        for(p=0;p<need;p+=4)if(s_readback[p]||s_readback[p+1]||s_readback[p+2])nz++;
        fprintf(stderr,"[NV2A-GL] sync %llu surface=0x%08X %ux%u nonblack=%llu/%llu GL=0x%04X\n",
                (unsigned long long)seq,surface_va,s->width,s->height,
                (unsigned long long)nz,(unsigned long long)((size_t)s->width*s->height),e);
        if(seq>=120 && nz==0){
            uint64_t first=s_draw_calls>8?s_draw_calls-8:0, n;
            fprintf(stderr,"[NV2A-GL] black-frame draw trace (%llu total draws):\n",
                    (unsigned long long)s_draw_calls);
            for(n=first;n<s_draw_calls;n++){
                const DrawTrace *d=&s_draw_trace[n%8];
                if(d->seq==n)fprintf(stderr,"[NV2A-GL]   draw %llu surface=%08X verts=%u xy=%.2f..%.2f/%.2f..%.2f z=%.3f..%.3f rhw=%.4g..%.4g rgba=%08X tex=%08X/%08X (%ux%u fmt=%02X pitch=%u) stage=%08X rc=%u ctrl=%08X comb=%08X/%08X:%08X/%08X final=%08X/%08X alpha=%u blend=%u:%X/%X depth=%u:%X write=%u cull=%u:%02X/%02X clip=%08X mask=%08X\n",
                    (unsigned long long)d->seq,d->surface,d->count,
                    d->min_x,d->max_x,d->min_y,d->max_y,d->min_z,d->max_z,d->min_rhw,d->max_rhw,d->diffuse,
                    d->tex0,d->tex1,d->tex0_size>>16,d->tex0_size&0xffffu,d->tex0_format,d->tex0_pitch,
                    d->stage,d->combiner,d->control,d->ci0,d->ai0,d->co0,d->ao0,d->final0,d->final1,d->alpha_test,
                    d->blend,d->blend_src,d->blend_dst,d->depth,d->depth_func,d->depth_write,d->cull,d->cull_face,d->front_face,d->clip_mode,d->color_mask);
            }
        }
    }
    dst=(uint8_t*)((uintptr_t)xbox_GetMemoryOffset()+s->color_va);
    if(s->bytes_per_pixel==4){for(y=0;y<s->height;y++)memcpy(dst+(size_t)y*s->pitch,s_readback+(size_t)y*s->width*4,(size_t)s->width*4);}
    else if(s->bytes_per_pixel==2){for(y=0;y<s->height;y++){uint32_t x;uint16_t *d=(uint16_t*)(dst+(size_t)y*s->pitch);const uint8_t *p=s_readback+(size_t)y*s->width*4;for(x=0;x<s->width;x++)d[x]=(uint16_t)(((p[x*4+2]>>3)<<11)|((p[x*4+1]>>2)<<5)|(p[x*4]>>3));}}
    if(!s_owns_window){
        int w=0,h=0;SDL_GL_GetDrawableSize(s_window,&w,&h);
        if(w>0&&h>0){
            /* The NV2A screen-space Y coordinate has its origin at the top,
             * while OpenGL's framebuffer origin is at the bottom. Flip the
             * source Y endpoints to match the guest-memory scanout. Also
             * restore state the preceding guest draw may have left behind. */
            glDisable(GL_SCISSOR_TEST);glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);
            glBindFramebuffer(GL_READ_FRAMEBUFFER,t->fbo);glReadBuffer(GL_COLOR_ATTACHMENT0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);glDrawBuffer(GL_BACK);
            glBlitFramebuffer(0,(GLint)s->height,(GLint)s->width,0,0,0,w,h,GL_COLOR_BUFFER_BIT,GL_NEAREST);
            pvideo_present(t,w,h);
            SDL_GL_SwapWindow(s_window);
        }
    }
}

int nv2a_gl_backend_init(void)
{
    static Nv2aBackend backend={backend_clear,backend_draw,backend_flip};GLuint vs,fs;GLint ok=0;
    if(!(SDL_WasInit(SDL_INIT_VIDEO)&SDL_INIT_VIDEO) && SDL_InitSubSystem(SDL_INIT_VIDEO)!=0){fprintf(stderr,"[NV2A-GL] SDL video init failed: %s\n",SDL_GetError());return 0;}
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,3);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
    s_window=(SDL_Window*)xbox_FramebufferWindowOpenGLWindow();
    if(s_window)s_owns_window=0;
    else{
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,0);
        s_window=SDL_CreateWindow("UC2 NV2A OpenGL",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,16,16,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
        s_owns_window=1;
    }
    if(!s_window){fprintf(stderr,"[NV2A-GL] GPU window creation failed: %s\n",SDL_GetError());return 0;}
    s_context=SDL_GL_CreateContext(s_window);if(!s_context){fprintf(stderr,"[NV2A-GL] GPU context creation failed: %s\n",SDL_GetError());if(s_owns_window)SDL_DestroyWindow(s_window);s_window=NULL;return 0;}
    if(SDL_GL_MakeCurrent(s_window,s_context)!=0){fprintf(stderr,"[NV2A-GL] GPU context activation failed: %s\n",SDL_GetError());return 0;}
    fprintf(stderr,"[NV2A-GL] hardware OpenGL %s | %s\n",glGetString(GL_VERSION),glGetString(GL_RENDERER));
    if(!glGetString(GL_RENDERER)||strstr((const char*)glGetString(GL_RENDERER),"llvmpipe")||strstr((const char*)glGetString(GL_RENDERER),"softpipe")){fprintf(stderr,"[NV2A-GL] software OpenGL renderer rejected; install/enable a hardware GPU driver\n");return 0;}
    vs=compile_shader(GL_VERTEX_SHADER,vs_src);fs=compile_shader(GL_FRAGMENT_SHADER,fs_src);if(!vs||!fs)return 0;
    s_program=glCreateProgram();glAttachShader(s_program,vs);glAttachShader(s_program,fs);glLinkProgram(s_program);glGetProgramiv(s_program,GL_LINK_STATUS,&ok);glDeleteShader(vs);glDeleteShader(fs);if(!ok){fprintf(stderr,"[NV2A-GL] shader link failed\n");return 0;}
    vs=compile_shader(GL_VERTEX_SHADER,pvideo_vs_src);fs=compile_shader(GL_FRAGMENT_SHADER,pvideo_fs_src);if(!vs||!fs)return 0;
    s_pvideo_program=glCreateProgram();glAttachShader(s_pvideo_program,vs);glAttachShader(s_pvideo_program,fs);glLinkProgram(s_pvideo_program);glGetProgramiv(s_pvideo_program,GL_LINK_STATUS,&ok);glDeleteShader(vs);glDeleteShader(fs);if(!ok){fprintf(stderr,"[NV2A-GL] PVIDEO shader link failed\n");return 0;}
    s_pv_base=glGetUniformLocation(s_pvideo_program,"u_base");s_pv_yuy=glGetUniformLocation(s_pvideo_program,"u_yuy");
    s_pv_window=glGetUniformLocation(s_pvideo_program,"u_window");s_pv_guest=glGetUniformLocation(s_pvideo_program,"u_guest");
    s_pv_origin=glGetUniformLocation(s_pvideo_program,"u_origin");s_pv_out_size=glGetUniformLocation(s_pvideo_program,"u_out_size");
    s_pv_in_size=glGetUniformLocation(s_pvideo_program,"u_in_size");s_pv_start=glGetUniformLocation(s_pvideo_program,"u_start");
    s_pv_step=glGetUniformLocation(s_pvideo_program,"u_step");s_pv_key_on=glGetUniformLocation(s_pvideo_program,"u_key_on");s_pv_key=glGetUniformLocation(s_pvideo_program,"u_key");
    {const GLfloat quad[12]={-1,-1,1,-1,1,1,-1,-1,1,1,-1,1};
     glGenVertexArrays(1,&s_pvideo_vao);glGenBuffers(1,&s_pvideo_vbo);glBindVertexArray(s_pvideo_vao);glBindBuffer(GL_ARRAY_BUFFER,s_pvideo_vbo);
     glBufferData(GL_ARRAY_BUFFER,sizeof quad,quad,GL_STATIC_DRAW);glEnableVertexAttribArray(0);glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(GLfloat),(void*)0);
     glGenTextures(1,&s_pvideo_texture);glBindTexture(GL_TEXTURE_2D,s_pvideo_texture);
     glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);}
    s_size=glGetUniformLocation(s_program,"u_size");s_alpha_enable=glGetUniformLocation(s_program,"u_alpha_enable");s_alpha_func=glGetUniformLocation(s_program,"u_alpha_func");s_alpha_ref=glGetUniformLocation(s_program,"u_alpha_ref");
    s_rc_enable=glGetUniformLocation(s_program,"u_rc_enable");s_rc_ci=glGetUniformLocation(s_program,"u_ci[0]");s_rc_ai=glGetUniformLocation(s_program,"u_ai[0]");s_rc_co=glGetUniformLocation(s_program,"u_co[0]");s_rc_ao=glGetUniformLocation(s_program,"u_ao[0]");s_rc_f0=glGetUniformLocation(s_program,"u_f0[0]");s_rc_f1=glGetUniformLocation(s_program,"u_f1[0]");
    s_rc_final0=glGetUniformLocation(s_program,"u_final0");s_rc_final1=glGetUniformLocation(s_program,"u_final1");s_rc_finalc0=glGetUniformLocation(s_program,"u_finalc0");s_rc_finalc1=glGetUniformLocation(s_program,"u_finalc1");s_rc_control=glGetUniformLocation(s_program,"u_control");s_rc_stage=glGetUniformLocation(s_program,"u_stage");s_rc_clip=glGetUniformLocation(s_program,"u_clip");
    for(int i=0;i<4;i++){char name[32];snprintf(name,sizeof name,"u_tex[%d]",i);s_tex_samplers[i]=glGetUniformLocation(s_program,name);snprintf(name,sizeof name,"u_tex_enabled[%d]",i);s_tex_enabled[i]=glGetUniformLocation(s_program,name);}
    glGenVertexArrays(1,&s_vao);glGenBuffers(1,&s_vbo);nv2a_backend_register_with_surface_sync(&backend,backend_sync);
    if(!s_owns_window){
        const char *vsync=getenv("RECOMP_PRESENT_VSYNC");
        int interval=(!vsync||!*vsync||*vsync!='0')?1:0;
        if(SDL_GL_SetSwapInterval(interval)!=0)fprintf(stderr,"[NV2A-GL] swap interval %d unavailable: %s\n",interval,SDL_GetError());
        fprintf(stderr,"[NV2A-GL] presenting render targets directly to the framebuffer window\n");
    }
    /* Leave it unowned; the executor thread will bind it on its first call. */
    SDL_GL_MakeCurrent(s_window,NULL);
    return 1;
}

void nv2a_gl_backend_shutdown(void)
{
    unsigned i;nv2a_backend_register(NULL);if(!s_context)return;
    if(SDL_GL_GetCurrentContext()!=s_context && SDL_GL_MakeCurrent(s_window,s_context)!=0)return;
    for(i=0;i<TEX_CACHE;i++)if(s_textures[i].name)glDeleteTextures(1,&s_textures[i].name);
    if(s_pvideo_texture)glDeleteTextures(1,&s_pvideo_texture);
    if(s_pvideo_vbo)glDeleteBuffers(1,&s_pvideo_vbo);
    if(s_pvideo_vao)glDeleteVertexArrays(1,&s_pvideo_vao);
    if(s_pvideo_program)glDeleteProgram(s_pvideo_program);
    for(i=0;i<TARGET_CACHE;i++)destroy_target(&s_targets[i]);
    if(s_vbo)glDeleteBuffers(1,&s_vbo);
    if(s_vao)glDeleteVertexArrays(1,&s_vao);
    if(s_program)glDeleteProgram(s_program);
    free(s_readback);
    free(s_vertices);s_vertices=NULL;s_vertex_capacity=0;
    SDL_GL_DeleteContext(s_context);if(s_owns_window)SDL_DestroyWindow(s_window);s_context=NULL;s_window=NULL;s_owns_window=0;
}
