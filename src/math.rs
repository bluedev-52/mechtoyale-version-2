use serde::Deserialize;
use std::f64::consts::{PI, TAU};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
pub struct V2(pub f64, pub f64);

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
pub struct V3(pub f64, pub f64, pub f64);

macro_rules! vector_ops {
    ($v:ident, $($i:tt),+) => {
        impl Add for $v {
            type Output = Self;
            fn add(self, rhs: Self) -> Self { Self($(self.$i + rhs.$i),+) }
        }
        impl Sub for $v {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self { Self($(self.$i - rhs.$i),+) }
        }
        impl Mul<f64> for $v {
            type Output = Self;
            fn mul(self, k: f64) -> Self { Self($(self.$i * k),+) }
        }
        impl Div<f64> for $v {
            type Output = Self;
            fn div(self, k: f64) -> Self { Self($(self.$i / k),+) }
        }
        impl Neg for $v {
            type Output = Self;
            fn neg(self) -> Self { self * -1.0 }
        }
        impl $v {
            pub fn dot(self, rhs: Self) -> f64 { 0.0 $(+ self.$i * rhs.$i)+ }
            pub fn norm2(self) -> f64 { self.dot(self) }
            pub fn norm(self) -> f64 { self.norm2().sqrt() }
            pub fn unit(self) -> Self { self / self.norm().max(1e-9) }
        }
    };
}

vector_ops!(V2, 0, 1);
vector_ops!(V3, 0, 1, 2);

impl V2 {
    pub fn cross(self, b: Self) -> f64 {
        self.0 * b.1 - self.1 * b.0
    }
    pub fn yaw(self) -> f64 {
        self.1.atan2(self.0)
    }
    pub fn perp(self) -> Self {
        Self(-self.1, self.0)
    }
    pub fn at(yaw: f64) -> Self {
        Self(yaw.cos(), yaw.sin())
    }
    pub fn z(self, z: f64) -> V3 {
        V3(self.0, self.1, z)
    }
}

impl V3 {
    pub fn xy(self) -> V2 {
        V2(self.0, self.1)
    }
    pub fn up(self, height: f64) -> Self {
        Self(self.0, self.1, self.2 + height)
    }
    pub fn aim(self) -> [f64; 2] {
        [self.xy().yaw(), self.2.atan2(self.xy().norm())]
    }
    pub fn dir([yaw, pitch]: [f64; 2]) -> Self {
        (V2::at(yaw) * pitch.cos()).z(pitch.sin())
    }
}

pub fn wrap_pi(angle: f64) -> f64 {
    (angle + PI).rem_euclid(TAU) - PI
}

pub struct SplitMix64(pub u64);
impl SplitMix64 {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut n = self.0;
        n = (n ^ (n >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        n = (n ^ (n >> 27)).wrapping_mul(0x94d049bb133111eb);
        n ^ (n >> 31)
    }
}
