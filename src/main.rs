use glam::{ vec2, Vec2, vec3, Vec3 };

use minifb::{ Window, WindowOptions, Key };

use smallvec::{ SmallVec, smallvec };

use rayon::prelude::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 960;

const RAYS_PER_UPDATE: usize = 750;
const EPS_ANGLE: f32 = 0.0001;
const HIT_THRESHOLD: f32 = 0.0001;
const NORMAL_EPS: f32 = 0.005;
const MAX_RAY_DEPTH: usize = 5;
const MAX_RAY_MARCHES: usize = 500;
const MAX_RAY_DISTANCE: f32 = 2000.0;
const MAX_ANGLE: f32 = 2.0 * (2.0 * std::f32::consts::PI);

fn setup_world() -> World {

    let red_diffuse = Lambertian {
        ior: 0.5,
        color: vec3(0.7, 0.2, 0.3),
    };
    let blue_diffuse = Lambertian {
        ior: 0.5,
        color: vec3(0.5, 0.2, 0.9),
    };

    let objects: Vec<Box<dyn WorldObject>> = vec![
        Box::new(Circle {
            center: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 4.0),
            radius: HEIGHT as f32 / 8.0,
            material: Box::new(red_diffuse.clone()),
            uuid: 0
        }),
        Box::new(Circle {
            center: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 - HEIGHT as f32 / 4.0),
            radius: HEIGHT as f32 / 8.0,
            material: Box::new(red_diffuse.clone()),
            uuid: 0
        }),
    ];

    let lights: Vec<Light> = vec![
        Light {
            pos: vec2(WIDTH as f32 / 2.0, HEIGHT as f32 / 2.0),
            spectrum: vec3(1.0, 0.4, 0.2),
            angle: 0.0,
        },
        Light {
            pos: vec2(WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(0.5, 0.4, 1.2),
            angle: 0.0,
        },
        Light {
            pos: vec2(WIDTH as f32 - WIDTH as f32 / 8.0, HEIGHT as f32 / 8.0),
            spectrum: vec3(1.0, 1.5, 0.6),
            angle: 0.0,
        },
    ];

    World { objects, lights }
}

#[derive(Clone, Copy, Debug)]
struct Lambertian {
    pub ior: f32,
    pub color: Vec3,
}

impl Material for Lambertian {
    fn evaluate_brdf(&self, ray_spectrum: Vec3, _hit_pos: Vec2, _incident: Vec2, _normal: Vec2, _out: Vec2) -> Vec3 {
        ray_spectrum * self.color
    }

    fn get_ior(&self) -> f32 {
        self.ior
    }
}

struct Circle {
    pub center: Vec2,
    pub radius: f32,
    pub material: Box<dyn Material + Send + Sync>,
    pub uuid: u64,
}

impl<'a> WorldObject for Circle {
    fn evaluate_brdf(&self, ray_spectrum: Vec3, hit_pos: Vec2, incident: Vec2, normal: Vec2, out: Vec2) -> Vec3 {
        self.material.evaluate_brdf(ray_spectrum, hit_pos, incident, normal, out)
    }

    fn get_ior(&self) -> f32 {
        self.material.get_ior()
    }

    fn evaluate_distance(&self, from: Vec2) -> f32 {
        (from - self.center).length() - self.radius
    }
    
    fn get_uuid(&self) -> u64 {
        self.uuid
    }
}

fn main() {
    let mut display_buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut img_buf: Vec<Vec3> = vec![vec3(0.0, 0.0, 0.0); WIDTH * HEIGHT];

    let mut window = Window::new(
        "rayn2d",
        WIDTH,
        HEIGHT,
        WindowOptions::default()).unwrap();

    let mut world = setup_world();
    
    //let mut draw_instructions: Vec<DrawInstruction> = Vec::with_capacity(2usize.pow(MAX_RAY_DEPTH as u32) * RAYS_PER_UPDATE * world.lights.len());
    let mut draw_instructions: Vec<DrawInstruction> = Vec::new();
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        update(&mut world, &mut draw_instructions);

        draw(&draw_instructions, &mut img_buf);
        draw_instructions.clear();

        update_display(&mut img_buf, &mut display_buf);

        window.update_with_buffer(&display_buf).unwrap();
    }
}

fn draw(draw_instructions: &[DrawInstruction], image_buf: &mut Vec<Vec3>) {
    for instruction in draw_instructions {
        draw_line(image_buf, instruction.p1, instruction.p2, instruction.spectrum);
    }
}

fn update_display(img_buf: &Vec<Vec3>, display_buf: &mut Vec<u32>) {
    for (img_pixel, display_pixel) in img_buf.iter().zip(display_buf.iter_mut()) {
        *display_pixel = vec3_to_u32(*img_pixel);
    }
}

fn update(world: &mut World, draw_instructions: &mut Vec<DrawInstruction>) {
    for light in world.lights.iter() {
        if light.angle > MAX_ANGLE {
            return;
        }
        draw_instructions.par_extend(
            rayon::iter::repeat(light).zip(0..RAYS_PER_UPDATE).flat_map(|(light, ray_number)| {
                let angle = light.angle + ray_number as f32 * EPS_ANGLE;
                if angle > MAX_ANGLE {
                    return Vec::new().into_par_iter()
                }
                let ray = Ray::new(light.pos, vec2(angle.cos(), angle.sin()), light.spectrum, 1.0);

                trace(world, ray).into_par_iter()
            })
        );
        // draw_instructions.extend(
        //     std::iter::repeat(light).zip(0..RAYS_PER_UPDATE).flat_map(|(light, ray_number)| {
        //         let angle = light.angle + ray_number as f32 * EPS_ANGLE;
        //         if angle > std::f32::consts::PI * 2.0 {
        //             return Vec::new().into_iter()
        //         }
        //         let ray = Ray::new(light.pos, vec2(angle.cos(), angle.sin()).normalize(), light.spectrum, 1.0);

        //         trace(world, ray).into_iter()
        //     })
        // );
    }

    for light in world.lights.iter_mut() {
        light.angle += RAYS_PER_UPDATE  as f32 * EPS_ANGLE;
    }
}

fn trace(world: &World, ray: Ray) -> Vec<DrawInstruction> {
    let mut draw_instructions = Vec::new();

    trace_inner(world, ray, &mut draw_instructions);

    draw_instructions
}

fn trace_inner(world: &World, mut ray: Ray, draw_instructions: &mut Vec<DrawInstruction>) {
    let mut total_marches = 0;
    let mut prev_dist = 0.0;

    loop {
        let (dist, closest) = dist_to_scene(world, ray.get_pos());
        if prev_dist == 0.0 {
            prev_dist = dist;
        }

        ray.advance(dist.abs());

        if dist.abs() < HIT_THRESHOLD {
            let hit_pos = ray.get_pos();

            draw_instructions.push(DrawInstruction {
                p1: ray.origin,
                p2: hit_pos,
                spectrum: ray.spectrum * EPS_ANGLE * 10.0,
            });

            if ray.depth < MAX_RAY_DEPTH {
                let relation = if prev_dist > 0.0 {
                    Relation::Outside
                } else {
                    Relation::Inside
                };
                let normal = estimate_normal(world, hit_pos, relation);

                let (reflection_ray, refraction_ray) = Ray::split_from(
                    &ray,
                    normal,
                    closest,
                    relation);

                //println!("reflection dir: {}, refraction dir: {}", reflection_ray.dir, refraction_ray.as_ref().unwrap().dir);
                trace_inner(world, reflection_ray, draw_instructions);

                if let Some(refraction_ray) = refraction_ray {
                    trace_inner(world, refraction_ray, draw_instructions);
                }
            }
            
            break;
        }

        total_marches += 1;
        prev_dist = dist;

        if ray.len >= MAX_RAY_DISTANCE || total_marches >= MAX_RAY_MARCHES {
            draw_instructions.push(DrawInstruction {
                p1: ray.origin,
                p2: ray.get_pos(),
                spectrum: ray.spectrum * EPS_ANGLE * 10.0,
            });

            break;
        }
    }
}

fn dist_to_scene(world: &World, p: Vec2) -> (f32, &dyn WorldObject) {
    let mut closest: Option<&dyn WorldObject> = None;
    let mut dist = std::f32::MAX;
    for object in world.objects.iter() {
        let obj_dist = object.evaluate_distance(p);
        if obj_dist < dist {
            dist = obj_dist;
            closest = Some(object.as_ref());
        }
    }
    (dist, closest.unwrap())
}

fn estimate_normal(world: &World, p: Vec2, relation: Relation) -> Vec2 {
  let x_p = dist_to_scene(world, vec2(p.x() + NORMAL_EPS, p.y())).0;
  let x_m = dist_to_scene(world, vec2(p.x() - NORMAL_EPS, p.y())).0;
  let y_p = dist_to_scene(world, vec2(p.x(), p.y() + NORMAL_EPS)).0;
  let y_m = dist_to_scene(world, vec2(p.x(), p.y() - NORMAL_EPS)).0;
  let x_diff = x_p - x_m;
  let y_diff = y_p - y_m;
  let vec = vec2(x_diff, y_diff);
  return vec.normalize() * if let Relation::Inside = relation { -1.0 } else { 1.0 };
}

struct World {
    pub objects: Vec<Box<dyn WorldObject>>,
    pub lights: Vec<Light>,
}

trait Material {
    fn evaluate_brdf(&self, ray_spectrum: Vec3, hit_pos: Vec2, incident: Vec2, normal: Vec2, out: Vec2) -> Vec3;
    fn get_ior(&self) -> f32;
}

trait WorldObject: Send + Sync {
    fn evaluate_distance(&self, from: Vec2) -> f32;
    fn evaluate_brdf(&self, ray_spectrum: Vec3, hit_pos: Vec2, incident: Vec2, normal: Vec2, out: Vec2) -> Vec3;
    fn get_ior(&self) -> f32;
    fn get_uuid(&self) -> u64;
}

#[derive(Clone, Copy, Debug)]
struct Light {
    pos: Vec2,
    spectrum: Vec3,
    angle: f32,
}

#[derive(Copy, Clone, Debug)]
struct DrawInstruction {
    p1: Vec2,
    p2: Vec2,
    spectrum: Vec3,
}

#[derive(Clone, Copy)]
pub enum MediumChange {
    Entering(f32, u64),
    Exiting(u64),
    Same,
}

#[derive(Clone, Copy)]
pub enum Relation {
    Outside,
    Inside,
}

#[derive(Clone)]
struct Ray {
    pub origin: Vec2,
    pub dir: Vec2,
    pub len: f32,
    pub spectrum: Vec3,
    pub depth: usize,
    pub medium_ior_stack: SmallVec<[(f32, u64); MAX_RAY_DEPTH]>,
}

impl Ray {
    fn new(origin: Vec2, dir: Vec2, spectrum: Vec3, medium_ior: f32) -> Ray { 
        Ray {
            origin,
            dir,
            len: 0f32,
            spectrum,
            depth: 0,
            medium_ior_stack: smallvec![(medium_ior, std::u64::MAX)],
        }
    }

    fn split_from(
        ray: &Ray,
        normal: Vec2,
        closest: &dyn WorldObject,
        closest_relation: Relation
    ) -> (Ray, Option<Ray>) {
        let reflection_medium_ior_stack = ray.medium_ior_stack.clone();

        let mut refraction_medium_ior_stack = ray.medium_ior_stack.clone();

        let refraction_dir = match closest_relation {
            Relation::Outside => {
                let uuid = closest.get_uuid();
                if refraction_medium_ior_stack.last().unwrap().1 != uuid {
                    let new_ior = closest.get_ior();
                    let old_ior = refraction_medium_ior_stack.last().unwrap().0;

                    refraction_medium_ior_stack.push((new_ior, uuid));

                    let eta = new_ior / old_ior;
                    ray.refract(normal, eta)
                } else {
                    None
                }
            },
            Relation::Inside => {
                let uuid = closest.get_uuid();
                if refraction_medium_ior_stack.last().unwrap().1 == uuid {
                    let old_ior = refraction_medium_ior_stack.pop().unwrap().0;
                    let new_ior = refraction_medium_ior_stack.last().unwrap().0;
                    
                    let eta = new_ior / old_ior;
                    ray.refract(normal, eta)
                } else {
                    None
                }
            },
        };

        let reflection_dir = ray.reflect(normal);

        let reflection_ray = Ray {
            origin: ray.get_pos() + 3.0 * HIT_THRESHOLD * normal,
            dir: reflection_dir,
            len: 0.0,
            spectrum: closest.evaluate_brdf(ray.spectrum, ray.get_pos(), ray.dir, normal, reflection_dir),
            //spectrum: vec3(0.0, 0.0, 0.0),
            depth: ray.depth + 1,
            medium_ior_stack: reflection_medium_ior_stack,
        };

        let refraction_ray = refraction_dir.map(|dir| {
            Ray {
                origin: ray.get_pos() - 3.0 * HIT_THRESHOLD * normal,
                dir,
                len: 0.0,
                spectrum: closest.evaluate_brdf(ray.spectrum, ray.get_pos(), ray.dir, normal, dir),
                // spectrum: vec3(2.0, 0.0, 0.0),
                depth: ray.depth + 1,
                medium_ior_stack: refraction_medium_ior_stack,
            }
        });

        (reflection_ray, refraction_ray)
    }

    fn advance(&mut self, dist: f32) { self.len += dist }
    fn get_pos(&self) -> Vec2 { self.origin + self.dir * self.len }

    fn reflect(&self, normal: Vec2) -> Vec2 {
        self.dir - 2.0 * self.dir.dot(normal) * normal
    }

    fn refract(&self, normal: Vec2, eta: f32) -> Option<Vec2> {
        let n_d_i = normal.dot(self.dir);
        let k = 1.0 - eta * eta * (1.0 - n_d_i * n_d_i);
        if k < 0.0 {
            None
        } else {
            Some(eta * self.dir - (eta * n_d_i + k.sqrt()) * normal)
        }
    }
}

// Line drawing algorithm

fn plot(buf: &mut Vec<Vec3>, x: i32, y: i32, a: f64, c: Vec3) {
    if x >= WIDTH as i32 || x < 0 || y >= HEIGHT as i32 || y < 0 {
        return;
    }
    
    let final_col = a as f32 * c;

    let pixel = &mut buf[x as usize + y as usize * WIDTH];

    *pixel += final_col;
}

fn vec3_to_u32(vec: Vec3) -> u32 {
    let r = (vec.x() * 255.0).max(0.0).min(255.0) as u32;
    let g = (vec.y() * 255.0).max(0.0).min(255.0) as u32;
    let b = (vec.z() * 255.0).max(0.0).min(255.0) as u32;
    return r << 16 | g << 8 | b;
}

fn ipart(x: f64) -> i32 {
    x as i32
}

fn fpart(x: f64) -> f64 {
    x - x.floor()
}
 
fn rfpart(x: f64) -> f64 {
    1.0 - fpart(x)
}
 
fn draw_line(buf: &mut Vec<Vec3>, p1: Vec2, p2: Vec2, s: Vec3) {
    let mut x0 = p1.x() as f64;
    let mut x1 = p2.x() as f64;
    let mut y0 = p1.y() as f64;
    let mut y1 = p2.y() as f64;
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    if steep {
      let mut t = x0;
      x0 = y0;
      y0 = t;
      t = x1;
      x1 = y1;
      y1 = t;
    }
 
    if x0 > x1 {
      let mut t = x0;
      x0 = x1;
      x1 = t;
      t = y0;
      y0 = y1;
      y1 = t;
    }
 
    let dx = x1 - x0;
    let dy = y1 - y0;
    let gradient = dy / dx;
    
    let angle = dy.atan2(dx);

    let m = 1.0 + 0.5 * (-(2.0 * angle).cos() + 1.0);

    // handle first endpoint
    let mut xend = (x0).round();
    let mut yend = y0 + gradient * (xend - x0);
    let mut xgap = rfpart(x0 + 0.5);
    let xpxl1 = xend as i32; // this will be used in the main loop
    let ypxl1 = ipart(yend);
 
    if steep {
        plot(buf, ypxl1, xpxl1, rfpart(yend) * xgap * m, s);
        plot(buf, ypxl1 + 1, xpxl1, fpart(yend) * xgap * m, s);
    } else {
        plot(buf, xpxl1, ypxl1, rfpart(yend) * xgap * m, s);
        plot(buf, xpxl1, ypxl1 + 1, fpart(yend) * xgap * m, s);
    }
 
    // first y-intersection for the main loop
    let mut intery = yend + gradient;
 
    // handle second endpoint
    xend = (x1).round();
    yend = y1 + gradient * (xend - x1);
    xgap = fpart(x1 + 0.5);
    let xpxl2 = xend as i32; // this will be used in the main loop
    let ypxl2 = ipart(yend);
 
    if steep {
        plot(buf, ypxl2, xpxl2, rfpart(yend) * xgap * m, s);
        plot(buf, ypxl2 + 1, xpxl2, fpart(yend) * xgap * m, s);
    } else {
        plot(buf, xpxl2, ypxl2, rfpart(yend) * xgap * m, s);
        plot(buf, xpxl2, ypxl2 + 1, fpart(yend) * xgap * m, s);
    }
 
    // main loop
    for x in (xpxl1 as i32 + 1)..(xpxl2 as i32 - 1) {
        if steep {
            plot(buf, ipart(intery), x, rfpart(intery) * m, s);
            plot(buf, ipart(intery) + 1, x, fpart(intery) * m, s);
        } else {
            plot(buf, x, ipart(intery), rfpart(intery) * m, s);
            plot(buf, x, ipart(intery) + 1, fpart(intery) * m, s);
        }
        intery = intery + gradient;
    }
}
