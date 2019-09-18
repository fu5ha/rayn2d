use glam::{ Vec2, Vec3 };

pub trait Material {
    fn evaluate_brdf(&self, ray_spectrum: Vec3, hit_pos: Vec2, incident: Vec2, normal: Vec2, out: Vec2) -> Vec3;
    fn get_ior(&self) -> f32;
}

#[derive(Clone, Copy, Debug)]
pub struct Lambertian {
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
