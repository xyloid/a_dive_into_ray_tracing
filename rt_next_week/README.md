# Ray Tracing: The Next Week

[source](https://raytracing.github.io/books/RayTracingTheNextWeek.html)

## 1. Motion Blur

In a real camera, the shutter opens and stays for a time interval, and the camera and objects may move during that time. Its really an average of what the camera sees over that interval that we want.

### Intuition of Ray Tracing

- visual quality worth more than run-time
- almost all effects can be brute-forced

### Introduction to SpaceTime Ray Tracing

generate rays at random times while the shutter is open and intersect the model at that one time.

The way it is usually done is to have the camera move and the objects move, but have each ray exist at exactly one time.

**Solution**

- add time to each ray, the ray exists at the time.
- modify the camera, it will only generate rays in an interval
    - the interval can be set by constructor parameters, so the function call is simpler, this is a personal preference on design choice.
- add moving object, the object will have a function that maps time to the position. I think it can be any trajectory generate method.
    - in the `hit` function, each time it will generate a new random center based on the interval given in the constructor.
- track the time in the intersection. Set time for scattered ray, reflected ray and refracted ray.

### Debug `cudaFree error 700`

[ref 1](https://stackoverflow.com/questions/58902166/why-do-i-have-insufficient-buffer-space-when-i-put-allocation-code-in-a-functi)
[ref 2](https://stackoom.com/question/3z98w/%E5%B0%86%E5%88%86%E9%85%8D%E4%BB%A3%E7%A0%81%E6%94%BE%E5%9C%A8%E5%87%BD%E6%95%B0%E4%B8%AD%E6%97%B6-%E4%B8%BA%E4%BB%80%E4%B9%88%E6%88%91%E7%9A%84%E7%BC%93%E5%86%B2%E5%8C%BA%E7%A9%BA%E9%97%B4%E4%B8%8D%E8%B6%B3)
```
Warning: 5 records have invalid timestamps due to insufficient device buffer space. You can configure the buffer space using the option --device-buffer-size.
```

cudaFree 700 error solved, the reason it casting moving_sphere to sphere in the free function

## 2. Bounding Volume Hierarchies

