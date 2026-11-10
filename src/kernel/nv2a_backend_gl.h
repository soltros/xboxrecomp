#ifndef XBOXRECOMP_NV2A_BACKEND_GL_H
#define XBOXRECOMP_NV2A_BACKEND_GL_H

/* Linux OpenGL NV2A rasterizer. Must be initialized on the guest/render
 * thread, after the host SDL video subsystem has been initialized. */
int nv2a_gl_backend_init(void);
void nv2a_gl_backend_shutdown(void);

#endif
