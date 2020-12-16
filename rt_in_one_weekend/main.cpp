#include "camera.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

#include <iostream>
#include <thread>
// using std::thread;

// bool hit_sphere(const point3 &center, double radius, const ray &r)
// {
//     vec3 ac = r.origin() - center;
//     auto a = dot(r.direction(), r.direction());
//     auto b = 2.0 * dot(ac, r.direction());
//     auto c = dot(ac, ac) - radius * radius;
//     auto discriminant = b * b - 4 * a * c;
//     return (discriminant > 0);
// }

double hit_sphere(const point3 &center, double radius, const ray &r) {
  vec3 ac = r.origin() - center;
  // auto a = dot(r.direction(), r.direction());
  auto a = r.direction().length_squared();
  // auto b = 2.0 * dot(ac, r.direction());
  auto half_b = dot(ac, r.direction());
  auto c = dot(ac, ac) - radius * radius;
  auto discriminant = half_b * half_b - a * c;
  if (discriminant < 0) {
    return -1.0;
  } else {
    // return (-b - sqrt(discriminant)) / (2.0 * a);
    return (-half_b - sqrt(discriminant)) / a;
  }
}

color ray_color(const ray &r) {
  auto t = hit_sphere(point3(0, 0, -1), 0.5, r);
  if (t > 0.0) {
    // surface-normal = surface point - center
    vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));

    // map surface normal to color
    return 0.5 * color(N.x() + 1, N.y() + 1, N.z() + 1);
  }

  vec3 unit_direction = unit_vector(r.direction());
  // scale [-1.0, 1.0] to [0, 1.0]
  t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

color ray_color(const ray &r, const hittable &world, int depth) {
  if (depth <= 0) {
    return color(0, 0, 0);
  }

  hit_record rec;

  // 0.001 to solve shadow acne problem, it seems lighter after this operation,
  // and performance improved
  if (world.hit(r, 0.001, infinity, rec)) {
    // // return 0.5 * (rec.normal + color(1, 1, 1));
    // // point3 target = rec.p + rec.normal + random_in_unit_sphere();
    // // True Lambertian Reflection
    // point3 target = rec.p + rec.normal + random_unit_vector();
    // // ray goes into next direction. when will it stop the recursive call?
    // return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    ray scattered;
    color attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation * ray_color(scattered, world, depth - 1);
    }
    return color(0, 0, 0);
  }
  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

// final scene
hittable_list random_scene() {
  hittable_list world;

  auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
  world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      auto choose_mat = random_double();
      point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

      if ((center - point3(4, 0.2, 0)).length() > 0.9) {
        shared_ptr<material> sphere_material;

        if (choose_mat < 0.8) {
          // diffuse
          auto albedo = color::random() * color::random();
          sphere_material = make_shared<lambertian>(albedo);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        } else if (choose_mat < 0.95) {
          // metal
          auto albedo = color::random(0.5, 1);
          auto fuzz = random_double(0, 0.5);
          sphere_material = make_shared<metal>(albedo, fuzz);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        } else {
          // glass
          sphere_material = make_shared<dielectric>(1.5);
          world.add(make_shared<sphere>(center, 0.2, sphere_material));
        }
      }
    }
  }

  // add three big spheres
  auto material1 = make_shared<dielectric>(1.5);
  world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

  auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
  world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

  auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
  world.add(make_shared<sphere>(color(4, 1, 0), 1.0, material3));

  return world;
}

void final_scene() {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  const int image_width = 800; // 1200
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 50; // 500
  const int max_depth = 50;

  // World
  auto world = random_scene();

  // Camera
  point3 lookfrom(13, 2, 3);
  point3 lookat(0, 0, 0);
  vec3 vup(0, 1, 0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;

  // Camera
  camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

  // Render
  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int j = image_height - 1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; s++) {
        auto u = (i + random_double()) / (image_width - 1);
        auto v = (j + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
      }
      write_color(std::cout, pixel_color, samples_per_pixel);

      /**
      auto u = double(i) / (image_width - 1);
      auto v = double(j) / (image_height - 1);

      ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
      color pixel_color = ray_color(r, world);

      write_color(std::cout, pixel_color);
      **/
    }
  }

  std::cerr << "\nDone.\n";
}

void learn() {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 800;
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 100;
  const int max_depth = 50;

  // World
  auto R = cos(pi / 4);
  hittable_list world;
  //   world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
  //   // any reason for this 100.5 ?
  //   world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));
  auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
  // auto material_center = make_shared<lambertian>(color(0.7, 0.3, 0.3));
  // auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
  // auto material_center = make_shared<dielectric>(1.8);
  auto material_center = make_shared<lambertian>(color(0.1, 0.2, 0.5));
  auto material_left = make_shared<dielectric>(1.5);
  auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);

  world.add(make_shared<sphere>(point3(0, -100.5, -1.0), 100, material_ground));
  world.add(make_shared<sphere>(point3(0, 0, -1.0), 0.5, material_center));
  world.add(make_shared<sphere>(point3(-1.0, 0, -1.0), 0.5, material_left));
  world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.4, material_left));
  world.add(make_shared<sphere>(point3(1.0, 0, -1.0), 0.5, material_right));
  // Camera
  point3 lookfrom(3, 3, 2);
  point3 lookat(0, 0, -1);
  vec3 vup(0, 1, 0);
  auto dist_to_focus = (lookfrom - lookat).length();
  auto aperture = 0.5;

  // camera cam(point3(-2, 2, 1), point3(0, 0, -1), vec3(0, 1, 0), 35.0,
  //            aspect_ratio);

  camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

  /**
   auto viewport_height = 2.0;
   auto viewport_width = aspect_ratio * viewport_height;
   auto focal_length = 1.0;

   auto origin = point3(0, 0, 0);
   auto horizontal = vec3(viewport_width, 0, 0);
   auto vertical = vec3(0, viewport_height, 0);
   auto lower_left_corner =
       origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);
  **/

  // Render
  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int j = image_height - 1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
      color pixel_color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; s++) {
        auto u = (i + random_double()) / (image_width - 1);
        auto v = (j + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
      }
      write_color(std::cout, pixel_color, samples_per_pixel);

      /**
      auto u = double(i) / (image_width - 1);
      auto v = double(j) / (image_height - 1);

      ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
      color pixel_color = ray_color(r, world);

      write_color(std::cout, pixel_color);
      **/
    }
  }

  std::cerr << "\nDone.\n";
}

// https://stackoverflow.com/questions/61985888/why-the-compiler-complains-that-stdthread-arguments-must-be-invocable-after-co

void worker(int start, int end,
            std::reference_wrapper<std::vector<shared_ptr<color>>> img,
            int image_width, int image_height, hittable_list world, camera cam,
            int samples_per_pixel, int max_depth) {
  std::cerr << start << "-" << end << std::endl;
  // [start, end)
  for (int index = start; index < end; index++) {
    int j = index / image_width;
    int i = index % image_width;
    color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
      auto u = (i + random_double()) / (image_width - 1);
      auto v = (j + random_double()) / (image_height - 1);
      ray r = cam.get_ray(u, v);
      pixel_color += ray_color(r, world, max_depth);
      // std::cerr << pixel_color << std::endl;
    }
    img.get().at(index) =
        make_shared<color>(pixel_color.x(), pixel_color.y(), pixel_color.z());
    // img.at(index) =
    //     make_shared<color>(pixel_color.x(), pixel_color.y(), pixel_color.z());
  }
}

void parallel_render() {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  const int image_width = 1200; // 1200
  const int image_height = static_cast<int>(image_width / aspect_ratio);
  const int samples_per_pixel = 500; // 500
  const int max_depth = 50;

  // World
  auto world = random_scene();

  // Camera
  point3 lookfrom(13, 2, 3);
  point3 lookat(0, 0, 0);
  vec3 vup(0, 1, 0);
  auto dist_to_focus = 10.0;
  auto aperture = 0.1;

  // Camera
  camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

  int size = image_height * image_width;

  std::cerr << "total size:" << size << std::endl;
  // color img[size];
  std::vector<shared_ptr<color>> img(size);
  int concurrency = 16;

  int batch_size = ceil(size / (double)concurrency);
  int last_batch = size % concurrency == 0 ? 0 : size % concurrency;

  std::vector<shared_ptr<std::thread>> tasks;
  for (int i = 0; i < concurrency; i++) {
    int start = batch_size * i;
    int end = std::min(batch_size * (i + 1), size);

    // std::thread thr(worker, start, end, img);
    tasks.push_back(make_shared<std::thread>(
        worker, start, end, std::ref(img), image_width, image_height, world,
        cam, samples_per_pixel, max_depth));
  }

  for (const auto t : tasks) {
    t->join();
  }

  std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  for (int j = image_height - 1; j >= 0; --j) {
    std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i = 0; i < image_width; ++i) {
      // std::cerr << img[j * image_width + i].get(). << std::endl;
      color pixel_color(img[j * image_width + i].get()->x(),
                        img[j * image_width + i].get()->y(),
                        img[j * image_width + i].get()->z());
      write_color(std::cout, pixel_color, samples_per_pixel);
    }
  }

  std::cerr << "\nDone.\n";

  std::cerr << "finished" << std::endl;
}

int main() {
  // learn();
  // final_scene();
  parallel_render();
}