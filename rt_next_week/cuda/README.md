# CUDA Implementation

## `shared_ptr` functions in CUDA device 

```
motion_blur.cu(36): error: calling a constexpr __host__ function("std::shared_ptr<    ::material> ::shared_ptr") from a __device__ function("get_color") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.

motion_blur.cu(36): error: identifier "std::shared_ptr<    ::material> ::shared_ptr" is undefined in device code

motion_blur.cu(40): error: calling a __host__ function("std::__shared_ptr_access<    ::material, ( ::__gnu_cxx::_Lock_policy)2, (bool)0, (bool)0> ::operator -> const") from a __device__ function("get_color") is not allowed

```

RAII pattern for CUDA [cuda smart pointer](https://stackoverflow.com/questions/16509414/is-there-a-cuda-smart-pointer)

An implementation of cuda shared_ptr [link](https://github.com/roostaiyan/CudaSharedPtr)

An old implementation of cpp std lib in cuda [ECUDA](https://baderlab.github.io/ecuda/)


## A Very Interesting "Bug"

When using `-G` option, BVH actually works and improved the speed.

But when I remove `-G`, there will be an cudaError 700 bug. 

[might be useful](https://forums.developer.nvidia.com/t/different-results-when-using-gpu-debug-option-g/30063/3)
[opposite example](https://stackoverflow.com/questions/14903063/what-is-the-granularity-of-the-cuda-memory-checker)

### ptxas warning

The two output below shows that when complie with `-G` option, there is no stack size problem in sort-related functions.

However, without `-G`, we have more memory leaks, I guess that's why we have cudaError 700/719 problem.

```
%nvcc  motion_blur.cu -o obj

ptxas warning : Stack size for entry function '_ZN6thrust8cuda_cub4core13_kernel_agentINS0_12__merge_sort14BlockSortAgentIPP8hittableS7_lPFbPKS5_S9_ENS_6detail17integral_constantIbLb0EEESE_EEbS7_S7_lS7_S7_SB_EEvT0_T1_T2_T3_T4_T5_T6_' cannot be statically determined
ptxas warning : Stack size for entry function '_ZN6thrust8cuda_cub4core13_kernel_agentINS0_12__merge_sort10MergeAgentIPP8hittableS7_lPFbPKS5_S9_ENS_6detail17integral_constantIbLb0EEEEEbS7_S7_lS7_S7_SB_PllEEvT0_T1_T2_T3_T4_T5_T6_T7_T8_' cannot be statically determined
ptxas warning : Stack size for entry function '_Z12create_worldPP8hittableS1_PP6cameraiiP17curandStateXORWOW' cannot be statically determined
ptxas warning : Stack size for entry function '_ZN6thrust8cuda_cub4core13_kernel_agentINS0_12__merge_sort14PartitionAgentIPP8hittablelPFbPKS5_S9_EEEbS7_S7_lmPlSB_liEEvT0_T1_T2_T3_T4_T5_T6_T7_T8_' cannot be statically determined
ptxas warning : Stack size for entry function '_Z6renderP4vec3iiiPP6cameraPP8hittableP17curandStateXORWOW' cannot be statically determined
```

```
nvcc -G motion_blur.cu -o obj

ptxas warning : Stack size for entry function '_Z12create_worldPP8hittableS1_PP6cameraiiP17curandStateXORWOW' cannot be statically determined
ptxas warning : Stack size for entry function '_Z6renderP4vec3iiiPP6cameraPP8hittableP17curandStateXORWOW' cannot be statically determined
```



[this warning means a memory leak](https://forums.developer.nvidia.com/t/is-it-important-to-fix-this-warning-message-ptxas-warning/79055)

### Temporary Solution

Since increasing the stack size solve the problem, I think the cause of the problem is the recursive call inside constructor of the bvh_node.

```
  cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);
```

### Performance for Motion Blur Rendering

This version of bvh must have something wrong.

With BVH

```
Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 575.416 seconds.
```

Without BVH

```
Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 123.074 seconds.
```

With BVH, Recursion to Iteration

```
Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 37.8792 seconds.
```

With BVH
```
==4321== NVPROF is profiling process 4321, command: ./obj
Rendering a 1200x800 image with 5 samples per pixel in 8x8 blocks.
took 2.98677 seconds.
==4321== Profiling application: ./obj
==4321== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.07%  2.98154s         1  2.98154s  2.98154s  2.98154s  render(vec3*, int, int, int, camera**, hittable**, curandStateXORWOW*)
                    3.38%  104.94ms         1  104.94ms  104.94ms  104.94ms  create_world(hittable**, hittable**, camera**, int, int, curandStateXORWOW*)
                    0.53%  16.413ms         1  16.413ms  16.413ms  16.413ms  free_world(hittable**, hittable**, camera**)
                    0.01%  449.33us         1  449.33us  449.33us  449.33us  render_init(int, int, curandStateXORWOW*)
                    0.00%  4.1280us         1  4.1280us  4.1280us  4.1280us  rand_init(curandStateXORWOW*)
      API calls:   92.78%  3.08700s         6  514.50ms  1.4090us  2.98156s  cudaDeviceSynchronize
                    4.94%  164.42ms         1  164.42ms  164.42ms  164.42ms  cudaDeviceSetLimit
                    1.11%  36.805ms         1  36.805ms  36.805ms  36.805ms  cudaDeviceReset
                    0.62%  20.561ms         1  20.561ms  20.561ms  20.561ms  cudaMallocManaged
                    0.52%  17.271ms         6  2.8785ms  3.7230us  16.498ms  cudaFree
                    0.01%  341.71us         5  68.341us  3.6340us  234.85us  cudaMalloc
                    0.01%  251.55us         1  251.55us  251.55us  251.55us  cuDeviceTotalMem
                    0.01%  208.25us         5  41.649us  6.6870us  138.70us  cudaLaunchKernel
                    0.00%  151.89us       101  1.5030us     141ns  65.839us  cuDeviceGetAttribute
                    0.00%  78.639us         1  78.639us  78.639us  78.639us  cuDeviceGetName
                    0.00%  8.6450us         1  8.6450us  8.6450us  8.6450us  cuDeviceGetPCIBusId
                    0.00%  5.8070us         2  2.9030us     160ns  5.6470us  cuDeviceGet
                    0.00%  1.7300us         3     576ns     192ns  1.3240us  cuDeviceGetCount
                    0.00%  1.5130us         6     252ns     144ns     324ns  cudaGetLastError
                    0.00%     280ns         1     280ns     280ns     280ns  cuDeviceGetUuid

==4321== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      96  117.21KB  4.0000KB  0.9961MB  10.98828MB  1.836850ms  Device To Host
      37         -         -         -           -  2.741589ms  Gpu page fault groups
Total CPU Page faults: 35

```

Without BVH

```
==4528== NVPROF is profiling process 4528, command: ./obj
Rendering a 1200x800 image with 5 samples per pixel in 8x8 blocks.
took 1.24292 seconds.
==4528== Profiling application: ./obj
==4528== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.19%  1.24037s         1  1.24037s  1.24037s  1.24037s  render(vec3*, int, int, int, camera**, hittable**, curandStateXORWOW*)
                    2.88%  37.133ms         1  37.133ms  37.133ms  37.133ms  create_world(hittable**, hittable**, camera**, int, int, curandStateXORWOW*)
                    0.89%  11.470ms         1  11.470ms  11.470ms  11.470ms  free_world(hittable**, hittable**, camera**)
                    0.04%  541.36us         1  541.36us  541.36us  541.36us  render_init(int, int, curandStateXORWOW*)
                    0.00%  4.0640us         1  4.0640us  4.0640us  4.0640us  rand_init(curandStateXORWOW*)
      API calls:   85.85%  1.27821s         6  213.03ms  2.2170us  1.24039s  cudaDeviceSynchronize
                    9.34%  139.11ms         1  139.11ms  139.11ms  139.11ms  cudaDeviceSetLimit
                    2.51%  37.364ms         1  37.364ms  37.364ms  37.364ms  cudaDeviceReset
                    1.38%  20.588ms         1  20.588ms  20.588ms  20.588ms  cudaMallocManaged
                    0.83%  12.352ms         6  2.0587ms  3.7500us  11.553ms  cudaFree
                    0.03%  511.71us         5  102.34us  5.9860us  342.73us  cudaMalloc
                    0.02%  270.96us         5  54.192us  5.6310us  179.03us  cudaLaunchKernel
                    0.01%  221.41us         1  221.41us  221.41us  221.41us  cuDeviceTotalMem
                    0.01%  130.35us       101  1.2900us     139ns  54.933us  cuDeviceGetAttribute
                    0.00%  72.455us         1  72.455us  72.455us  72.455us  cuDeviceGetName
                    0.00%  11.285us         1  11.285us  11.285us  11.285us  cuDeviceGetPCIBusId
                    0.00%  2.1850us         6     364ns     169ns     628ns  cudaGetLastError
                    0.00%  1.9430us         3     647ns     208ns  1.5090us  cuDeviceGetCount
                    0.00%     836ns         2     418ns     154ns     682ns  cuDeviceGet
                    0.00%     332ns         1     332ns     332ns     332ns  cuDeviceGetUuid

==4528== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      96  117.21KB  4.0000KB  0.9961MB  10.98828MB  1.868725ms  Device To Host
      33         -         -         -           -  3.343241ms  Gpu page fault groups
Total CPU Page faults: 35

```

### BVH Recursion to Iteration

[Thinking Parallel, Part II: Tree Traversal on the GPU](https://developer.nvidia.com/blog/thinking-parallel-part-ii-tree-traversal-gpu/)

```
Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 37.8792 seconds.
```

### Unplugged Laptop

Last night I finished the BVH part and it works fine on my laptop, the performance is as expected. 
However, this morning, when I run the BVH Iteration version, the runtime is `200 sec` not `39 sec`. I am
 surprised and suspect that my GPU might not working in high performance mode, the fan noise was mush less than 
 last night. Suddenly, I realized that my laptop was not plugged correctly, so it was using the battery and automatically switched to power-saving mode.