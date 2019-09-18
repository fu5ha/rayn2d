pub const WIDTH: usize = 1280;
pub const HEIGHT: usize = 720;

pub const RAYS_PER_SAMPLE: usize = 10_000;
pub const RAYS_PER_UPDATE: usize = 500;
pub const EXPORT_AT_SAMPLE_COUNT: usize = 5;

pub const HIT_THRESHOLD: f32 = 0.0001;
pub const NORMAL_EPS: f32 = 0.005;
pub const MAX_RAY_DEPTH: usize = 10;
pub const MAX_RAY_MARCHES: usize = 1000;
pub const MAX_RAY_DISTANCE: f32 = (WIDTH * 2) as f32;