# Triangles

## Goals

The final goal is simple, I hope the render can render tirangle meshed objects in the scene. And I listed several
subgoals below.

- Understand and implement triangle hittable class
    - `hit`
    - `bounding_box`
- Understand Wavefront obj file format
    - read and parse the file in an naive way
    - optimize using pthreads (optional)
    - optimize using cuda (optional)

## Debug

- Custom model with triangles incorrectly rendered
- Triangle could have zero width bounding box
    - adding thickness do help the triangle shows up


## Large number of triangles causing crash

```
========= CUDA-MEMCHECK
Rendering a 400x400 image with 10 samples per pixel in 8x8 blocks.
========= Out-of-range Shared or Local Address
=========     at 0x000000d0 in __cuda_syscall_mc_dyn_globallock_check
=========     by thread (0,4,0) in block (8,1,0)
=========     Device Frame:/home/xyl/projects/a_dive_into_ray_tracing/triangles/cuda/./include/camera.h:67:_ZNK6camera7get_rayEffP17curandStateXORWOW (_ZNK6camera7get_rayEffP17curandStateXORWOW : 0x3d0)
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 (cuLaunchKernel + 0x2b8) [0x222dc8]
=========     Host Frame:./obj [0x2f56b]
=========     Host Frame:./obj [0x75245]
=========     Host Frame:./obj [0x27114]
=========     Host Frame:./obj [0x26193]
=========     Host Frame:./obj [0x26205]
=========     Host Frame:./obj [0x257b9]
=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x270b3]
=========     Host Frame:./obj [0x8d4e]
=========
CUDA error = ========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
=========     Saved host backtrace up to driver entry point at error
=========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so.1 [0x37a373]
719=========     Host Frame:./obj [0x59799]
 at =========     Host Frame:./obj [0x257e1]
obj_render.cu:=========     Host Frame:/lib/x86_64-linux-gnu/libc.so.6 (__libc_start_main + 0xf3) [0x270b3]
771=========     Host Frame:./obj [0x8d4e]
 '=========
cudaDeviceSynchronize()' 
========= Error: process didn't terminate successfully
=========        The application may have hit an error when dereferencing Unified Memory from the host. Please rerun the application under cuda-gdb or Nsight Eclipse Edition to catch host side errors.
========= No CUDA-MEMCHECK results found
```