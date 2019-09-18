use crate::light::Light;

use glam::{ Vec2, Vec3 };

pub struct World {
    pub objects: Vec<Box<dyn WorldObject>>,
    pub lights: Vec<Box<dyn Light>>,
}

pub trait WorldObject: Send + Sync {
    fn evaluate_distance(&self, from: Vec2) -> f32;
    fn evaluate_brdf(&self, ray_spectrum: Vec3, hit_pos: Vec2, incident: Vec2, normal: Vec2, out: Vec2) -> Vec3;
    fn get_ior(&self) -> f32;
    fn get_uuid(&self) -> u64;
}

