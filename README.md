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

## Accelerated Ray Tracing with CUDA

```
prime-select nvidia

time ./obj > final.ppm
Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 146.189 seconds.
./obj > final.ppm  146.42s user 0.14s system 99% cpu 2:26.65 total

Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 129.241 seconds.
./obj > final.ppm  129.46s user 0.15s system 99% cpu 2:09.67 total


prime-select intel

Rendering a 1200x800 image with 500 samples per pixel in 8x8 blocks.
took 130.967 seconds.
./obj > final.ppm  131.20s user 0.14s system 99% cpu 2:11.39 total

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
    - Hard shadows: shooting one ray from a point light
    - Soft shadows: shooting a bunch of rays from an area light
    - light bouncing around: Interreflection / Indirect lighting / Color bleeding / Global illumination
    - Glossy reflections
    - Ambient occlusion
      - It has been used with rasterization for a long time.
      - With ray tracing, we can get a better answer.
    - Depth of field.
    - Motion blur
    - Ray marching (an atmospheric effect)
    - Caustics
      - light effect with water or other transparent media

  - [Ray Tracing Essentials Part 6: The Rendering Equation](https://www.youtube.com/watch?v=AODo_RjJoUA&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=6)
    - It sums up **how lights get into the eye**
    - How to sample the rays on one point in the scene
      - Random sampling uniformaly over a unit sphere
      - Add BSDF
      - Add ligithing

  - [Ray Tracing Essentials Part 7: Denoising for Ray Tracing](https://www.youtube.com/watch?v=6O2B9BZiZjQ&list=PL5B692fm6--sgm8Uiava0IIvUojjFOCSR&index=7)
    - using denoise will improve the performance.
      - less samples per pixel
    - start with a noisy result and reconstruct
    - many different approaches
      - denoising by effect
      - deep learning for image denoising

- [Ray Tracing Gems](http://www.realtimerendering.com/raytracinggems/)

- [Ray Tracing Gems II](http://www.realtimerendering.com/raytracinggems/rtg2/index.html) (Under construction)

- [Introduction to Realtime Ray Tracing](http://rtintro.realtimerendering.com/)

- [Ray Tracing Resource Page](http://www.realtimerendering.com/raytracing.html)

- [UCD: Depth Buffers and Ray Tracing](https://www.youtube.com/watch?v=Xks1v4GNUiY)
- [UCD: Ray Tracing](https://www.youtube.com/watch?v=Ahp6LDQnK4Y)

- [Introduction to Computer Graphics (fall 2018), Lecture 12: Accelerating Ray Tracing](https://www.youtube.com/watch?v=FbLCMy-M2ls)


### Camera and Lens

- [Pixar in a Box: Virtual Cameras](https://www.khanacademy.org/computing/pixar/virtual-cameras)

- [UCB CS184/284A](https://cs184.eecs.berkeley.edu/uploads/lectures/)
  - [Cameras & Lenses I](https://cs184.eecs.berkeley.edu/uploads/lectures/20_cameras-1/20_cameras-1_slides.pdf)
  - [Cameras & Lenses II](https://cs184.eecs.berkeley.edu/uploads/lectures/21_camera-2/21_camera-2_slides.pdf)
  - [Cameras & Lenses III](https://cs184.eecs.berkeley.edu/uploads/lectures/22_camera-3/22_camera-3_slides.pdf)
- [UW Thin Lenses](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwi5yvnE1-ntAhUBqlkKHfOPDcUQFjANegQIMBAC&url=https%3A%2F%2Fcanvas.uw.edu%2Ffiles%2F44759652%2Fdownload%3Fdownload_frd%3D1&usg=AOvVaw38WeHxWsNTNzU474RkSlGU)

- [Depth of Field in Path Tracing](https://medium.com/@elope139/depth-of-field-in-path-tracing-e61180417027)

### Coding

- [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
  - [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
  - [Accelerated Ray Tracing in One Weekend in CUDA](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/)
- [Ray Tracing: The Next Week](https://raytracing.github.io/books/RayTracingTheNextWeek.html) 
- [Ray Tracing: The Rest of Your Life](https://raytracing.github.io/books/RayTracingTheRestOfYourLife.html)

### A Note on Nsight Eclipse Plugin for CUDA 11

`com.spotify.docker.client` is no longer maintained, I have to download 2 jars manually and copy them to `$ECLISPE/dropins/plugins`. 
Then start eclipse with `eclipse -clean`, so the eclipse will clean the cache and load plugins.

- Dependencies
  - `com.fasterxml.jackson.datatype.jackson-datatype-guava_2.9.9.v20190906-1522.jar`
  - `com.spotify.docker.client_8.11.7.v20180731-1413.jar`
