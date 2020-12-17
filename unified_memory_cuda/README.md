# Unified Memory for CUDA Beginners
[source](https://developer.nvidia.com/blog/unified-memory-cuda-beginners/)

## An Interesting Problem from "An Even Easier Introduction to CUDA"

In the previous code, my Turing gpu was beaten by Kepler on average. It seems this article is about that problem. 

## Unified Memory

**Unified Memory** is **a single memory address space** accessible from **any processor in a system**, either CPU or GPU.

### How to allocate?

```
cudaError_t cudaMallocManaged(void** ptr, size_t size);
```

### Implementation

CUDA system sw and/or hw takes care of **migrating memory pages to the memory of the accessing processor**.

Pascal GPU architecture is the first with hardware support for **virtual memory page faulting and migration**, via its **Page Migration Engine**.

Kepler and Maxwell also support a more limited form of Unified Memory.

## Kepler and `cudaMallocManaged()`

```
cudaError_t cudaMallocManaged(void** ptr, size_t size);
```

It will allocate `size` of memory on the GPU device that is active when the call is made.

Internelly, the driver will also setup page table entries for all pages covered by the allocation, so that the system knows that the pages are resident on that GPU.

### Explain the Example

- arrays allocated in GPU memory
- initialization casues page fault when CPU try to write to both arrays.
- thus driver migrates the page from device memory to CPU memory.
- upon launching kernel `add`, the CUDA runtime will migrate all pages previously migrated to host memory or to another GPU back to the device memory of the device running the kernel. 
    - this is because the GPU can not page fault, all data must be resident on the GPU just in case the kernel accesses it (even if it won’t).
    - This means there is **potentially migration overhead** on each **kernel launch**.
 
## Pascal and `cudaMallocManaged()`

- managed memory may not be physically allocated when `cudaMallocManaged` **returns**. 
    - it may only be populated on access (or prefetching).
    - pages and page table may not be created until they are accessed by the GPU or CPU.
    - the pages can migrate to any processor's memory at any time, and the driver employs **heuristics** to maintain data locality and **pevent excessive page faults**.  
    - Applications can guide the driver using `cudaMemAdvise()`, and explicitly migrate memory using `cudaMemPrefetchAsync()`, as [this blog post describes](https://developer.nvidia.com/blog/parallelforall/beyond-gpu-memory-limits-unified-memory-pascal/).
- Kernel launches without any migration overhead.
    - When it accesses any absent pages, the GPU stalls execution of the accessing threads, and the Page Migration Engine migrates the pages to the device before resuming the threads.
    - This means the cost of the migratoins is included in the kernel running time.

## Solutions to Migration Overhead

In real application, the GPU is likely to perform a lot more computation on data(perhaps many times) without the CPU touching it. The migration overhead in this simple code is caused by the fact that the CPU initializes the data and the GPU only uses it once.

- Move the data initialization to the GPU in another CUDA kernel.
- Run the kernel many times and look at the average and minimum run times.
- Prefetch the data to GPU memory before running the kernel.
