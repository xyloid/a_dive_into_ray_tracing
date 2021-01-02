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