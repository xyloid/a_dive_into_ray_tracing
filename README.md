# A Dive into Ray Tracing

This project is inspired by Peter Shirley's Ray Tracing series.

## Contents
- Basics
  - Computer Graphics
  - C++ in Practice
- Hopefully
  - CUDA Acceleration
  - Perfromance Engineering

## Ray Tracing in One Weekend

```
1200 16:9 500 samples per pixel 50 max depth 
16 partitions

./inOneWeek > image_parallel.ppm  49831.04s user 333.87s system 1242% cpu 1:07:16.10 total
```

## References


### Background Knowledge

- [Ray Tracing Essentials from nVidia](https://www.youtube.com/playlist?list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR)
  - [Ray Tracing Essentials, Part 1: Basics of Ray Tracing](https://www.youtube.com/watch?v=gBPNO6ruevk)
    - view ray, shadow ray
    - do it from the eye, limited number of rays to cast
    - in film, 3000 rays per pixel, add up rays and approximate the right answer.
  - [Ray Tracing Essentials Part 2: Rasterization versus Ray Tracing](https://www.youtube.com/watch?v=ynCxnR1i0QY&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=2)
    - Rasterization (z-buffer) loop:
      ```
      for each object
        for each pixel -> closer ?
      ```
    - Ray Tracing loop:
      ```
      for each pixel
        for each object -> closest ?
      ```
    
    - Bounding Volume Hierarchy (BVH)
      - a tree structure for effecient rendering
      - model's triangles are divided by multiple boxes, so the ray only will check a subset of all triangles.

    - Rasterization and Ray Tracing can work together.
-  [Ray Tracing Essentials Part 3: Ray Tracing Hardware](https://www.youtube.com/watch?v=EoQfX1q-VNE&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=3)

- [Ray Tracing Essentials Part 4: The Ray Tracing Pipeline](https://www.youtube.com/watch?v=LoKUmbvbcRY&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=4)
  - Ray Tracing pipeline has 5 shaders:
    - Control other shaders:
      - Ray generation shader
    - Define object shapes:
      - Intersection shader(s)
    - Control per-ray behavior (often many types)
      - Miss shader(s)
      - Closest-hit shader(s)
      - Any-hit shader(s)


- [Ray Tracing Essentials Part 5: Ray Tracing Effects](https://www.youtube.com/watch?v=Rk5nD8tt_W4&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=5)



- [Ray Tracing Gems](http://www.realtimerendering.com/raytracinggems/)

- [Ray Tracing Gems II](http://www.realtimerendering.com/raytracinggems/rtg2/index.html) (Under construction)

- [Introduction to Realtime Ray Tracing](http://rtintro.realtimerendering.com/)

- [Ray Tracing Resource Page](http://www.realtimerendering.com/raytracing.html)


### Coding

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
  - [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  - [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
- [Ray Tracing: The Rest of Your Life](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html)
