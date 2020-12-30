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