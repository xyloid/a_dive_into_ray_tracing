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

- main bottle neck: calculation of ray-object intersection.
    - run time: O(# obj)
    - repeated search on the same model (what about animation ?)
    - should be able to use divide and conquer to archieve sub-linear search
- common approaches
    - divide the space
    - divide the objects (much easier to code up)

### Key Idea

**bounding volume** a volume that fully encloses (bounds) all the objects

```python
if (ray hits bounding object)
    return whether ray hits bounded objects 
else
    return False
```

- we are dividing objects into subsets.
- we are not dividing screen or the volume
- any object is in just one bounding volume
- bounding volumes can overlap

### Hierarchies of Bounding Volumes

- it is a tree
- the tree has no order
- subtrees can overlap, in the overlap volume, each object belongs to exactly one of the two bounding volumes

```python
if(hits purple)
    hit0 = hits blue enclosed objects
    hit1 = hits red enclosed objects
    if (hit0 or hit1)
        return True and info of closer hit
return False
```

### Axis-Aligned Bounding Boxes (AABBs)

Question: What are octree and k-d tree ?

[ref 1](https://www.gamedev.net/forums/topic/289728-octrees-vs-kd-trees/)

Factors of design bounding volume

- ray boundding volume intersection should be fast
- bounding volume should be compact
- What we want to know 
    - whether or not it's a hit ?
- What we dont want to know
    - hit points
    - hit normals
- AABB: axis-aligned bounding rectangular parallelepiped axis-aligned bounding box
- n-dimensional AABB: the intersection of n axis-aligned intervals, often called "slabs"

### Constructing Bounding Boxes for Hittables

- compute the bounding boxes of all the hittables
- make a hierarchy of boxes over all the primitives, and the individual primitives (like the sphere will live at the leaves)

### The BVH Node Class







