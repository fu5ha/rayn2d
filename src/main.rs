use glam::{ vec2, Vec2, vec3, Vec3 };

use minifb::{ Window, WindowOptions, Key };

use std::sync::mpsc::{ Sender, Receiver };

use rayon::prelude::*;

const WIDTH: usize = 1280;
const HEIGHT: usize = 960;

const RAYS_PER_UPDATE: usize = 500;
const DELTA_ANGLE: f32 = 0.0001;

fn main() {
    let mut display_buf: Vec<u32> = vec![0; WIDTH * HEIGHT];
    let mut img_buf: Vec<Vec3> = vec![Vec3::new(0.0, 0.0, 0.0); WIDTH * HEIGHT];

    let mut window = Window::new(
        "rayn2d",
        WIDTH,
        HEIGHT,
        WindowOptions::default()).unwrap();

    let objects: Vec<Box<dyn WorldObject + Send + Sync>> = vec![];
    let lights: Vec<Light> = vec![
        Light {
            pos: Vec2::new(0.0, 0.0),
            spectrum: Vec3::new(0.8, 2.0, 1.0),
            angle: 0.0,
        },
    ];
    
    let mut draw_instructions: Vec<DrawInstruction> = Vec::with_capacity(RAYS_PER_UPDATE * lights.len());

    let world = World { objects, lights };
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        update(&world, &mut draw_instructions);

        draw(&draw_instructions, &mut img_buf);
        draw_instructions.clear();

        update_display(&mut img_buf, &mut display_buf);

        window.update_with_buffer(&display_buf).unwrap();
    }
}

fn update_display(img_buf: &Vec<Vec3>, display_buf: &mut Vec<u32>) {
    for (img_pixel, display_pixel) in img_buf.iter().zip(display_buf.iter_mut()) {
        *display_pixel = vec3_to_u32(*img_pixel);
    }
}

fn update(world: &World, draw_instructions: &mut Vec<DrawInstruction>) {
    for light in world.lights {
        draw_instructions.par_extend(
            rayon::iter::repeat(light).zip(0..RAYS_PER_UPDATE).flat_map(|(light, ray_number)| {
                let angle = light.angle + ray_number as f32 * DELTA_ANGLE;
                let ray = Ray::new(light.pos, Vec2::new(angle.cos(), angle.sin()));

                trace(world, ray).into_par_iter()
            })
        );
    }

    for light in world.lights.iter_mut() {
        light.angle += RAYS_PER_UPDATE  as f32 * DELTA_ANGLE;
    }
}

fn trace(world: &World, ray: Ray) -> Vec<DrawInstruction> {
    let mut draw_instructions = Vec::new();

    trace_inner(world, ray, &mut draw_instructions, 0);

    draw_instructions
}

fn trace_inner(world: &World, ray: Ray, draw_instructions: &mut Vec<DrawInstruction>, bounces: usize) {
    while 
}

fn draw(draw_instructions: &[DrawInstruction], image_buf: &mut Vec<Vec3>) {
    for instruction in draw_instructions {
        draw_line(image_buf, instruction.p1, instruction.p2, instruction.spectrum);
    }
}

struct World {
    pub objects: Vec<Box<dyn WorldObject + Send + Sync>>,
    pub lights: Vec<Light>,
}

trait WorldObject {
    fn evaluate_distance(&self, from: Vec2) -> f32;
    fn evaluate_brdf(&self, ray_in_dir: Vec2, normal: Vec2) -> Vec3;
    fn get_ior(&self) -> f32;
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

unsafe impl Send for DrawInstruction {}

struct Ray {
    pub origin: Vec2,
    pub dir: Vec2,
    pub len: f32
}

impl Ray {
    fn new(origin: Vec2, dir: Vec2) -> Ray { Ray { origin, dir, len: 0f32 } }

    fn advance(&mut self, dist: f32) { self.len += dist }
    fn get_pos(&self) -> Vec2 { self.origin + self.dir * self.len }
}


// Line drawing algorithm

fn plot(buf: &mut Vec<Vec3>, x: i32, y: i32, a: f64, c: Vec3) {
    let final_col = a as f32 * c;

    let pixel = &mut buf[x as usize + y as usize * WIDTH];

    *pixel += c;
}

fn vec3_to_u32(vec: Vec3) -> u32 {
    let r = ((vec.x() * 255.0).max(0.0).min(255.0) as u32) << 16;
    let g = ((vec.x() * 255.0).max(0.0).min(255.0) as u32) << 8;
    let b = (vec.x() * 255.0).max(0.0).min(255.0) as u32;
    return r | g | b;
}

fn u32_to_vec3(c: u32) -> Vec3 {
    let r = (c >> 16) as f32 / 255.0;
    let g = (c & 0x00FF00 >> 8) as f32 / 255.0;
    let b = (c & 0x0000FF) as f32 / 255.0;

    Vec3::new(r, g, b)
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
    let intery = yend + gradient;
 
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
