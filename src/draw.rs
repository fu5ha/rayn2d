use crate::consts::*;

use glam::{ Vec2, Vec3 };

#[derive(Copy, Clone, Debug)]
pub struct DrawInstruction {
    pub p1: Vec2,
    pub p2: Vec2,
    pub spectrum: Vec3,
}

pub fn draw(draw_instructions: &[DrawInstruction], image_buf: &mut Vec<Vec3>) {
    for instruction in draw_instructions {
        draw_line(image_buf, instruction.p1, instruction.p2, instruction.spectrum);
    }
}

pub fn update_display(img_buf: &Vec<Vec3>, display_buf: &mut Vec<u32>) {
    for (img_pixel, display_pixel) in img_buf.iter().zip(display_buf.iter_mut()) {
        *display_pixel = vec3_to_u32(*img_pixel);
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