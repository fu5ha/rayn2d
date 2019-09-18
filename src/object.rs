use crate::world::*;
use crate::material::*;

use glam::{ Vec2, Vec3 };

pub struct Circle {
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