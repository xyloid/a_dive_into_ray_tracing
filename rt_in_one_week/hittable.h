#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

#include <memory>
#include <vector>

using std::make_shared;
using std::shared_ptr;

struct hit_record
{
    point3 p;
    vec3 normal;
    double t;
    bool front_face;

    // always face out side.
    inline void set_face_nromal(const ray &r, const vec3 &outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable
{
public:
    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const = 0;
};

class hittable_list : public hittable
{
public:
    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }
    void clear() { objects.clear(); }
    void add(shared_ptr<hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

public:
    std::vector<shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray &r, double t_min, double t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    auto closet_so_far = t_max;

    for (const auto &object : objects)
    {
        if (object->hit(r, t_min, closet_so_far, temp_rec))
        {
            hit_anything = true;
            closet_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

#endif