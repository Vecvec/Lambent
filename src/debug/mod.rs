use crate::low_level::{self, RayTracingShader};
use wesl::include_wesl;

/// Debugging shader to show whether the ray has hit the front face (green) or back face (red)
/// useful for translucent materials as they determine whether the ray is entering by whether
/// it hit the front face, to give some depth perception the brightness of the pixel is 1 / (depth + 1)
pub struct FrontFace;

unsafe impl RayTracingShader for FrontFace {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(
        _opts: &dyn low_level::RayTracerOptions,
    ) -> String {
        include_wesl!("front_face").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "front_face debugging shader"
    }
}

pub struct Reflectance;

unsafe impl RayTracingShader for Reflectance {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(
        _opts: &dyn low_level::RayTracerOptions,
    ) -> String {
        include_wesl!("reflectance").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "reflectance debugging shader"
    }
}

pub struct Tangent;

unsafe impl RayTracingShader for Tangent {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(
        _opts: &dyn low_level::RayTracerOptions,
    ) -> String {
        include_wesl!("tangent").to_string()
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "tangent debugging shader"
    }
}
