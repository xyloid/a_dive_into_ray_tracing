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

1. Move the data initialization to the GPU in another CUDA kernel.
2. Run the kernel many times and look at the average and minimum run times.
3. Prefetch the data to GPU memory before running the kernel.


## Tests

### Init in Kernel

```
==11039== NVPROF is profiling process 11039, command: ./add_grid_init
Max error: 0
==11039== Profiling application: ./add_grid_init
==11039== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   95.23%  1.0634ms         1  1.0634ms  1.0634ms  1.0634ms  init(int, float*, float*)
                    4.77%  53.280us         1  53.280us  53.280us  53.280us  add(int, float*, float*)
      API calls:   98.99%  197.01ms         2  98.507ms  44.075us  196.97ms  cudaMallocManaged
                    0.56%  1.1181ms         1  1.1181ms  1.1181ms  1.1181ms  cudaDeviceSynchronize
                    0.19%  371.04us         2  185.52us  75.156us  295.88us  cudaFree
                    0.12%  244.99us         1  244.99us  244.99us  244.99us  cuDeviceTotalMem
                    0.08%  164.36us       101  1.6270us     154ns  66.613us  cuDeviceGetAttribute
                    0.02%  48.078us         2  24.039us  7.9980us  40.080us  cudaLaunchKernel
                    0.02%  47.608us         1  47.608us  47.608us  47.608us  cuDeviceGetName
                    0.00%  6.2910us         1  6.2910us  6.2910us  6.2910us  cuDeviceGetPCIBusId
                    0.00%  1.4700us         3     490ns     235ns     974ns  cuDeviceGetCount
                    0.00%     881ns         2     440ns     177ns     704ns  cuDeviceGet
                    0.00%     317ns         1     317ns     317ns     317ns  cuDeviceGetUuid

==11039== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  664.9610us  Device To Host
      14         -         -         -           -  1.028770ms  Gpu page fault groups
Total CPU Page faults: 12
```

### Run Many Times

100 times

```
==12089== NVPROF is profiling process 12089, command: ./add_grid_many
Max error: 0
==12089== Profiling application: ./add_grid_many
==12089== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  8.1625ms       100  81.624us  53.344us  2.8587ms  add(int, float*, float*)
      API calls:   94.62%  161.19ms         2  80.595ms  66.728us  161.12ms  cudaMallocManaged
                    4.66%  7.9457ms         1  7.9457ms  7.9457ms  7.9457ms  cudaDeviceSynchronize
                    0.26%  440.02us         2  220.01us  200.50us  239.52us  cudaFree
                    0.19%  320.29us       100  3.2020us  2.7370us  27.657us  cudaLaunchKernel
                    0.13%  228.25us         1  228.25us  228.25us  228.25us  cuDeviceTotalMem
                    0.09%  154.72us       101  1.5310us     135ns  67.467us  cuDeviceGetAttribute
                    0.04%  72.467us         1  72.467us  72.467us  72.467us  cuDeviceGetName
                    0.01%  9.0580us         1  9.0580us  9.0580us  9.0580us  cuDeviceGetPCIBusId
                    0.00%  1.3890us         3     463ns     170ns     991ns  cuDeviceGetCount
                    0.00%     846ns         2     423ns     146ns     700ns  cuDeviceGet
                    0.00%     284ns         1     284ns     284ns     284ns  cuDeviceGetUuid

==12089== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     110  74.473KB  4.0000KB  984.00KB  8.000000MB  1.575782ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  662.9460us  Device To Host
      12         -         -         -           -  2.819880ms  Gpu page fault groups
Total CPU Page faults: 36
```

200 times

```
==12188== NVPROF is profiling process 12188, command: ./add_grid_many
Max error: 0
==12188== Profiling application: ./add_grid_many
==12188== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  13.437ms       200  67.184us  53.088us  2.8332ms  add(int, float*, float*)
      API calls:   91.03%  146.82ms         2  73.410ms  62.750us  146.76ms  cudaMallocManaged
                    8.06%  12.995ms         1  12.995ms  12.995ms  12.995ms  cudaDeviceSynchronize
                    0.39%  622.69us       200  3.1130us  2.6950us  32.550us  cudaLaunchKernel
                    0.27%  428.09us         2  214.04us  193.17us  234.92us  cudaFree
                    0.14%  221.77us         1  221.77us  221.77us  221.77us  cuDeviceTotalMem
                    0.08%  127.43us       101  1.2610us     140ns  54.339us  cuDeviceGetAttribute
                    0.04%  71.492us         1  71.492us  71.492us  71.492us  cuDeviceGetName
                    0.00%  6.4660us         1  6.4660us  6.4660us  6.4660us  cuDeviceGetPCIBusId
                    0.00%  1.8240us         3     608ns     203ns  1.3930us  cuDeviceGetCount
                    0.00%     861ns         2     430ns     157ns     704ns  cuDeviceGet
                    0.00%     377ns         1     377ns     377ns     377ns  cuDeviceGetUuid

==12188== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
     102  80.314KB  4.0000KB  972.00KB  8.000000MB  1.564066ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  662.8160us  Device To Host
      12         -         -         -           -  2.791588ms  Gpu page fault groups
Total CPU Page faults: 36
```

### Prefetch

```
==13567== NVPROF is profiling process 13567, command: ./add_grid_prefetch
Max error: 0
==13567== Profiling application: ./add_grid_prefetch
==13567== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  48.640us         1  48.640us  48.640us  48.640us  add(int, float*, float*)
      API calls:   97.91%  145.14ms         2  72.569ms  50.737us  145.09ms  cudaMallocManaged
                    0.97%  1.4306ms         1  1.4306ms  1.4306ms  1.4306ms  cudaDeviceSynchronize
                    0.46%  675.58us         2  337.79us  137.11us  538.47us  cudaMemPrefetchAsync
                    0.37%  547.76us         2  273.88us  243.75us  304.01us  cudaFree
                    0.15%  219.84us         1  219.84us  219.84us  219.84us  cuDeviceTotalMem
                    0.10%  140.99us       101  1.3950us     141ns  62.059us  cuDeviceGetAttribute
                    0.03%  45.035us         1  45.035us  45.035us  45.035us  cuDeviceGetName
                    0.02%  29.895us         1  29.895us  29.895us  29.895us  cudaLaunchKernel
                    0.00%  7.3250us         1  7.3250us  7.3250us  7.3250us  cuDeviceGetPCIBusId
                    0.00%  3.5720us         1  3.5720us  3.5720us  3.5720us  cudaGetDevice
                    0.00%  1.4440us         3     481ns     173ns  1.0860us  cuDeviceGetCount
                    0.00%     704ns         2     352ns     170ns     534ns  cuDeviceGet
                    0.00%     253ns         1     253ns     253ns     253ns  cuDeviceGetUuid

==13567== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
       4  2.0000MB  2.0000MB  2.0000MB  8.000000MB  1.317314ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  664.0680us  Device To Host
Total CPU Page faults: 36
```