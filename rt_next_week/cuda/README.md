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