# Accelerated Ray Tracing in One Weekend in CUDA
[source](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)

## Sections

1. First Image
    - move computation to devcie
2. Adding Vectors
    - create class used in both host and device
3. Classing Up the GPU
    - use class in device
4. Hit the Sphere
    - add function in device
5. Manage Your Memory
    - no `shared_ptr` or `make_shared` here, just plain cpp. 
    - But, there is something interesting: [C++11 smart pointers and CUDA](https://ernestyalumni.wordpress.com/2017/09/28/bringing-cuda-into-the-year-2011-c11-smart-pointers-with-cuda-cub-nccl-streams-and-cuda-unified-memory-management-with-cub-and-cublas/)
    - `__device__` can be applied to `virtual` functions.

## reference
- [Unified Memory for CUDA Beginners](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

- [inline functions](https://forums.developer.nvidia.com/t/inline-functions-not-inlined-in-cuda-6-5/35788/5)
