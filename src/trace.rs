use crate::world::*;
use crate::consts::*;
use crate::draw::DrawInstruction;

#[cfg(feature = "profile")]
use thread_profiler::profile_scope;

use glam::{ vec2, Vec2, Vec3 };

use smallvec::{ smallvec, SmallVec };

pub struct TracerState {
    pub current_ray_count: usize,
    pub ray_index_vec: Vec<usize>,
    pub current_sample_count: usize,
}

pub fn trace(world: &World, ray: Ray) -> Vec<DrawInstruction> {
    #[cfg(feature = "profile")]
    profile_scope!("trace");

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
                spectrum: ray.spectrum,
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
                spectrum: ray.spectrum,
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

#[derive(Clone, Copy)]
pub enum Relation {
    Outside,
    Inside,
}

#[derive(Clone)]
pub struct Ray {
    pub origin: Vec2,
    pub dir: Vec2,
    pub len: f32,
    pub spectrum: Vec3,
    pub depth: usize,
    pub medium_ior_stack: SmallVec<[(f32, u64); MAX_RAY_DEPTH]>,
}

impl Ray {
    pub fn new(origin: Vec2, dir: Vec2, spectrum: Vec3, medium_ior: f32) -> Ray { 
        Ray {
            origin,
            dir,
            len: 0f32,
            spectrum,
            depth: 0,
            medium_ior_stack: smallvec![(medium_ior, std::u64::MAX)],
        }
    }

    pub fn split_from(
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