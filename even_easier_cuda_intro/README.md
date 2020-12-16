# An Even Easier Introduction to CUDA

## Simple Add on CPU and GPU

```
1M elements:
Max error:0
./add  0.01s user 0.00s system 98% cpu 0.018 total

Max error: 0
./add_cuda  0.11s user 0.08s system 74% cpu 0.251 total

```
## Profile

```
==7930== NVPROF is profiling process 7930, command: ./add_cuda
Max error: 0
==7930== Profiling application: ./add_cuda
==7930== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  156.28ms         1  156.28ms  156.28ms  156.28ms  add(int, float*, float*)
      API calls:   51.70%  168.35ms         2  84.176ms  54.703us  168.30ms  cudaMallocManaged
                   48.00%  156.30ms         1  156.30ms  156.30ms  156.30ms  cudaDeviceSynchronize
                    0.13%  437.49us         2  218.75us  195.23us  242.26us  cudaFree
                    0.07%  238.83us         1  238.83us  238.83us  238.83us  cuDeviceTotalMem
                    0.05%  155.69us       101  1.5410us     141ns  68.501us  cuDeviceGetAttribute
                    0.02%  76.951us         1  76.951us  76.951us  76.951us  cuDeviceGetName
                    0.01%  32.403us         1  32.403us  32.403us  32.403us  cudaLaunchKernel
                    0.00%  15.740us         1  15.740us  15.740us  15.740us  cuDeviceGetPCIBusId
                    0.00%  5.1540us         2  2.5770us     155ns  4.9990us  cuDeviceGet
                    0.00%  1.8110us         3     603ns     194ns  1.4090us  cuDeviceGetCount
                    0.00%     288ns         1     288ns     288ns     288ns  cuDeviceGetUuid

==7930== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  1.421896ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  656.8220us  Device To Host
      12         -         -         -           -  2.820396ms  Gpu page fault groups
Total CPU Page faults: 36
```

With 256 blocks

```
==8525== NVPROF is profiling process 8525, command: ./add_block
Max error: 0
==8525== Profiling application: ./add_block
==8525== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  3.2557ms         1  3.2557ms  3.2557ms  3.2557ms  add(int, float*, float*)
      API calls:   97.93%  197.45ms         2  98.723ms  38.456us  197.41ms  cudaMallocManaged
                    1.62%  3.2656ms         1  3.2656ms  3.2656ms  3.2656ms  cudaDeviceSynchronize
                    0.22%  442.97us         2  221.49us  199.87us  243.10us  cudaFree
                    0.11%  219.03us         1  219.03us  219.03us  219.03us  cuDeviceTotalMem
                    0.07%  138.45us       101  1.3700us     141ns  59.216us  cuDeviceGetAttribute
                    0.03%  57.719us         1  57.719us  57.719us  57.719us  cuDeviceGetName
                    0.01%  27.378us         1  27.378us  27.378us  27.378us  cudaLaunchKernel
                    0.00%  9.6220us         1  9.6220us  9.6220us  9.6220us  cuDeviceGetPCIBusId
                    0.00%  4.2900us         2  2.1450us     174ns  4.1160us  cuDeviceGet
                    0.00%  1.6770us         3     559ns     206ns     938ns  cuDeviceGetCount
                    0.00%     271ns         1     271ns     271ns     271ns  cuDeviceGetUuid

==8525== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      48  170.67KB  4.0000KB  0.9961MB  8.000000MB  1.412790ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  658.4370us  Device To Host
      12         -         -         -           -  2.498116ms  Gpu page fault groups
Total CPU Page faults: 36
```

With grid

```
==14943== NVPROF is profiling process 14943, command: ./add_grid
Max error: 0
==14943== Profiling application: ./add_grid
==14943== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:  100.00%  2.8097ms         1  2.8097ms  2.8097ms  2.8097ms  add(int, float*, float*)
      API calls:   97.71%  162.50ms         2  81.248ms  50.300us  162.45ms  cudaMallocManaged
                    1.70%  2.8215ms         1  2.8215ms  2.8215ms  2.8215ms  cudaDeviceSynchronize
                    0.27%  449.55us         2  224.78us  200.95us  248.60us  cudaFree
                    0.15%  247.72us         1  247.72us  247.72us  247.72us  cuDeviceTotalMem
                    0.10%  162.18us       101  1.6050us     151ns  70.668us  cuDeviceGetAttribute
                    0.05%  77.881us         1  77.881us  77.881us  77.881us  cuDeviceGetName
                    0.02%  31.427us         1  31.427us  31.427us  31.427us  cudaLaunchKernel
                    0.01%  8.6960us         1  8.6960us  8.6960us  8.6960us  cuDeviceGetPCIBusId
                    0.00%  1.2540us         3     418ns     186ns     841ns  cuDeviceGetCount
                    0.00%     965ns         2     482ns     180ns     785ns  cuDeviceGet
                    0.00%     276ns         1     276ns     276ns     276ns  cuDeviceGetUuid

==14943== Unified Memory profiling result:
Device "GeForce RTX 2060 with Max-Q Design (0)"
   Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
      99  82.747KB  4.0000KB  972.00KB  8.000000MB  1.549740ms  Host To Device
      24  170.67KB  4.0000KB  0.9961MB  4.000000MB  663.4600us  Device To Host
      12         -         -         -           -  2.766326ms  Gpu page fault groups
Total CPU Page faults: 36
```