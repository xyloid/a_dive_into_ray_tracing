# Triangles

## Goals

The final goal is simple, I hope the render can render tirangle meshed objects in the scene. And I listed several
subgoals below.

- Understand and implement triangle hittable class
    - `hit`
    - `bounding_box`
- Understand Wavefront obj file format
    - read and parse the file in an naive way
    - optimize using pthreads (optional)
    - optimize using cuda (optional)

## Debug

- Custom model with triangles incorrectly rendered
- Triangle could have zero width bounding box
    - adding thickness do help the triangle shows up