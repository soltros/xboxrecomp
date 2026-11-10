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

typedef struct { Nv2aSurface s; GLuint fbo, color, depth; } Target;
typedef struct { Nv2aTexture key; GLuint name; } CachedTexture;
typedef struct { float x,y,z,rhw; uint8_t color[4]; float u,v; } GpuVertex;

static SDL_Window *s_window;
static SDL_GLContext s_context;
static GLuint s_program, s_vao, s_vbo;
static GLint s_size, s_sampler, s_use_texture, s_alpha_enable, s_alpha_func, s_alpha_ref;
static Target s_target;
static CachedTexture s_textures[TEX_CACHE];
static unsigned s_next_texture;
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
    "layout(location=2) in vec2 a_uv; uniform vec2 u_size;"
    "out vec4 v_color; out vec2 v_uv;"
    "void main(){ vec2 p=a_pos.xy/u_size*2.0-1.0;"
    " gl_Position=vec4(p,a_pos.z*2.0-1.0,1.0);"
    " v_color=a_color; v_uv=a_uv; }";
static const char *fs_src =
    "#version 330 core\n"
    "in vec4 v_color; in vec2 v_uv; out vec4 o_color;"
    "uniform sampler2D u_tex; uniform int u_use_texture;"
    "uniform int u_alpha_enable; uniform int u_alpha_func; uniform float u_alpha_ref;"
    "void main(){ vec4 c=v_color; if(u_use_texture!=0)c*=texture(u_tex,v_uv);"
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

static int ensure_target(const Nv2aSurface *s)
{
    if(s_target.fbo && s_target.s.color_va==s->color_va && s_target.s.width==s->width
       && s_target.s.height==s->height) return 1;
    if(s_target.fbo){ glDeleteFramebuffers(1,&s_target.fbo); glDeleteTextures(1,&s_target.color); glDeleteRenderbuffers(1,&s_target.depth); }
    memset(&s_target,0,sizeof s_target); s_target.s=*s;
    glGenTextures(1,&s_target.color); glBindTexture(GL_TEXTURE_2D,s_target.color);
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,(GLsizei)s->width,(GLsizei)s->height,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glGenFramebuffers(1,&s_target.fbo); glBindFramebuffer(GL_FRAMEBUFFER,s_target.fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,s_target.color,0);
    glGenRenderbuffers(1,&s_target.depth); glBindRenderbuffer(GL_RENDERBUFFER,s_target.depth);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8,(GLsizei)s->width,(GLsizei)s->height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_RENDERBUFFER,s_target.depth);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){ fprintf(stderr,"[NV2A-GL] framebuffer setup failed\n"); return 0; }
    return 1;
}

static int same_tex(const Nv2aTexture *a,const Nv2aTexture *b)
{ return a->offset==b->offset && a->width==b->width && a->height==b->height && a->color==b->color && a->pitch==b->pitch; }

static GLuint texture_for(const Nv2aTexture *t)
{
    uint32_t i; GLuint n; uint32_t *px;
    if(!t || !t->width || !t->height || (uint64_t)t->width*t->height>16777216u) return 0;
    for(i=0;i<TEX_CACHE;i++) if(s_textures[i].name && same_tex(&s_textures[i].key,t)) return s_textures[i].name;
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
    (void)rs; if(!make_current() || !ensure_target(s))return; glBindFramebuffer(GL_FRAMEBUFFER,s_target.fbo); glViewport(0,0,(GLsizei)s->width,(GLsizei)s->height);
    GLbitfield bits=0;
    if(flags&0xf0){glClearColor(((argb>>16)&255)/255.f,((argb>>8)&255)/255.f,(argb&255)/255.f,((argb>>24)&255)/255.f);bits|=GL_COLOR_BUFFER_BIT;}
    if(flags&1){glClearDepth(((zs>>8)&0xffffffu)/16777215.0);bits|=GL_DEPTH_BUFFER_BIT;}
    if(flags&2){glClearStencil(zs&255);bits|=GL_STENCIL_BUFFER_BIT;}
    glClear(bits);
}

static void backend_draw(const Nv2aSurface *s,const Nv2aBatch *b)
{
    uint32_t i; GpuVertex *v; GLuint tex=0; const Nv2aRenderState *r=b->state;
    if(!b->count || !make_current() || !ensure_target(s))return;
    v=(GpuVertex*)malloc((size_t)b->count*sizeof *v); if(!v)return;
    for(i=0;i<b->count;i++){const Nv2aVertex *a=&b->vertices[i]; v[i].x=a->x;v[i].y=a->y;v[i].z=a->z;v[i].rhw=a->rhw;
        v[i].color[0]=(uint8_t)(a->diffuse>>16);v[i].color[1]=(uint8_t)(a->diffuse>>8);v[i].color[2]=(uint8_t)a->diffuse;v[i].color[3]=(uint8_t)(a->diffuse>>24);
        v[i].u=a->u/(b->texture&&b->texture->width?b->texture->width:1);v[i].v=a->v/(b->texture&&b->texture->height?b->texture->height:1);}
    if(b->texture)tex=texture_for(b->texture);
    glBindFramebuffer(GL_FRAMEBUFFER,s_target.fbo);glViewport(0,0,(GLsizei)s->width,(GLsizei)s->height);glUseProgram(s_program);glBindVertexArray(s_vao);
    glEnableVertexAttribArray(0);glEnableVertexAttribArray(1);glEnableVertexAttribArray(2);
    glUniform2f(s_size,(float)s->width,(float)s->height);glUniform1i(s_sampler,0);glUniform1i(s_use_texture,tex?1:0);
    glUniform1i(s_alpha_enable,r->alpha_test_enable?1:0);glUniform1i(s_alpha_func,(GLint)r->alpha_func);glUniform1f(s_alpha_ref,r->alpha_ref/255.f);
    if(tex){glActiveTexture(GL_TEXTURE0);glBindTexture(GL_TEXTURE_2D,tex);}
    if(r->blend_enable){glEnable(GL_BLEND);glBlendFunc((GLenum)r->blend_src,(GLenum)r->blend_dst);glBlendEquation((GLenum)r->blend_eq);}else glDisable(GL_BLEND);
    if(r->depth_test_enable){glEnable(GL_DEPTH_TEST);glDepthFunc((GLenum)r->depth_func);}else glDisable(GL_DEPTH_TEST);glDepthMask(r->depth_write?GL_TRUE:GL_FALSE);
    glColorMask((r->color_mask&0x00010000u)!=0,(r->color_mask&0x100u)!=0,(r->color_mask&1)!=0,(r->color_mask&0x01000000u)!=0);
    if(r->cull_enable){glEnable(GL_CULL_FACE);glCullFace(r->cull_face==0x0404?GL_FRONT:GL_BACK);glFrontFace(r->front_face==0x0900?GL_CW:GL_CCW);}else glDisable(GL_CULL_FACE);
    if(r->stencil_enable){glEnable(GL_STENCIL_TEST);glStencilFunc((GLenum)r->stencil_func,(GLint)r->stencil_ref,r->stencil_func_mask);glStencilMask(r->stencil_write_mask);glStencilOp((GLenum)r->stencil_fail,(GLenum)r->stencil_zfail,(GLenum)r->stencil_zpass);}else glDisable(GL_STENCIL_TEST);
    glBindBuffer(GL_ARRAY_BUFFER,s_vbo);glBufferData(GL_ARRAY_BUFFER,(GLsizeiptr)((size_t)b->count*sizeof *v),v,GL_STREAM_DRAW);
    glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,sizeof *v,(void*)offsetof(GpuVertex,x));
    glVertexAttribPointer(1,4,GL_UNSIGNED_BYTE,GL_TRUE,sizeof *v,(void*)offsetof(GpuVertex,color));
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof *v,(void*)offsetof(GpuVertex,u));
    glDrawArrays(GL_TRIANGLES,0,(GLsizei)b->count);free(v);
}

static void backend_flip(void) { if(make_current())glFlush(); }

static void backend_sync(void)
{
    size_t need; uint32_t y; uint8_t *dst; const Nv2aSurface *s=&s_target.s;
    if(!make_current() || !s_target.fbo || !s->color_va || !s->width || !s->height)return;
    glBindFramebuffer(GL_FRAMEBUFFER,s_target.fbo);glFinish();need=(size_t)s->width*s->height*4;
    if(need>s_readback_size){uint8_t *p=(uint8_t*)realloc(s_readback,need);if(!p)return;s_readback=p;s_readback_size=need;}
    glPixelStorei(GL_PACK_ALIGNMENT,1);glReadPixels(0,0,(GLsizei)s->width,(GLsizei)s->height,GL_BGRA,GL_UNSIGNED_BYTE,s_readback);
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
    s_size=glGetUniformLocation(s_program,"u_size");s_sampler=glGetUniformLocation(s_program,"u_tex");s_use_texture=glGetUniformLocation(s_program,"u_use_texture");s_alpha_enable=glGetUniformLocation(s_program,"u_alpha_enable");s_alpha_func=glGetUniformLocation(s_program,"u_alpha_func");s_alpha_ref=glGetUniformLocation(s_program,"u_alpha_ref");
    glGenVertexArrays(1,&s_vao);glGenBuffers(1,&s_vbo);nv2a_backend_register_with_present_sync(&backend,backend_sync);
    /* Leave it unowned; the executor thread will bind it on its first call. */
    SDL_GL_MakeCurrent(s_window,NULL);
    return 1;
}

void nv2a_gl_backend_shutdown(void)
{
    unsigned i;nv2a_backend_register(NULL);if(!s_context)return;
    if(SDL_GL_GetCurrentContext()!=s_context && SDL_GL_MakeCurrent(s_window,s_context)!=0)return;
    for(i=0;i<TEX_CACHE;i++)if(s_textures[i].name)glDeleteTextures(1,&s_textures[i].name);
    if(s_target.fbo)glDeleteFramebuffers(1,&s_target.fbo);
    if(s_target.color)glDeleteTextures(1,&s_target.color);
    if(s_target.depth)glDeleteRenderbuffers(1,&s_target.depth);
    if(s_vbo)glDeleteBuffers(1,&s_vbo);
    if(s_vao)glDeleteVertexArrays(1,&s_vao);
    if(s_program)glDeleteProgram(s_program);
    free(s_readback);
    SDL_GL_DeleteContext(s_context);SDL_DestroyWindow(s_window);s_context=NULL;s_window=NULL;
}
