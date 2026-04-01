use std::num::NonZeroU32;

use wesl::include_wesl;

use crate::low_level::{self, RayTracingShader};

pub struct PathTracerOpts {
    pub samples: NonZeroU32,
    pub restir_temporal: RestirTemporalMode,
}

impl Default for PathTracerOpts {
    fn default() -> Self {
        Self { samples: low_level::DEFAULT_NUM_SAMPLES, restir_temporal: Default::default() }
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub enum RestirTemporalMode {
    /// Warning: if spatial resampling is used it will break
    Disabled,
    /// Restir is applied per sample, joined with resamples, per pixel.
    PerSample,
    /// Restir is applied per pixel.
    #[default]
    PerPixel,
}

impl low_level::RayTracerOptions for PathTracerOpts {
    fn samples_ref(&self) -> &NonZeroU32 {
        &self.samples
    }

    fn get_non_sample<'a>(&'a self, name: &'static str) -> &'a dyn std::any::Any {
        if name == "restir_temporal" {
            &self.restir_temporal
        } else {
            &()
        }
    }
}

fn map_handler(handler: RestirTemporalMode) -> &'static str {
    match handler {
        RestirTemporalMode::Disabled => "const RESTIR:u32 = RESTIR_OPTION_NONE;",
        RestirTemporalMode::PerSample => "const RESTIR:u32 = RESTIR_OPTION_PER_SAMPLE;",
        RestirTemporalMode::PerPixel => "const RESTIR:u32 = RESTIR_OPTION_PER_PIXEL;",
    }
}

/// A ray-tracing shader, note that this requires ~6-10x the samples of the [Medium] shader.
///
/// Differences from [Medium]
///  - Picks a wavelength of light and uses that as its colour
pub struct High;

unsafe impl RayTracingShader for High {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(opts: &dyn low_level::RayTracerOptions) -> String {
        let ray_tracing_processor = map_handler(low_level::RayTracerOptions::get(opts, "restir_temporal").downcast_ref::<RestirTemporalMode>().map_or_else(RestirTemporalMode::default, |v| *v));

        include_wesl!("high_path_tracing").to_string() + ray_tracing_processor
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing High shader"
    }
}

/// A ray-tracing shader
///
/// Features
///  - Randomly picks refractive index between low and high refractive indices
///  - 2D roughness for surface and 2D roughness for coating.
pub struct Medium;

unsafe impl RayTracingShader for Medium {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(opts: &dyn low_level::RayTracerOptions) -> String {
        let ray_tracing_processor = map_handler(low_level::RayTracerOptions::get(opts, "restir_temporal").downcast_ref::<RestirTemporalMode>().map_or_else(RestirTemporalMode::default, |v| *v));

        include_wesl!("medium_path_tracing").to_string() + ray_tracing_processor
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing Medium shader"
    }
}

/// A ray-tracing shader, note that this only needs ~0.9x the samples of the [Medium] shader.
///
/// Differences from [Medium]
///  - Ignores roughness
///  - Only uses one refractive index
pub struct Low;

unsafe impl RayTracingShader for Low {
    fn new() -> Self {
        Self
    }
    fn shader_source_without_intersection_handler(opts: &dyn low_level::RayTracerOptions) -> String {
        let ray_tracing_processor = map_handler(low_level::RayTracerOptions::get(opts, "restir_temporal").downcast_ref::<RestirTemporalMode>().map_or_else(RestirTemporalMode::default, |v| *v));

        include_wesl!("low_path_tracing").to_string() + ray_tracing_processor
    }
    #[cfg(debug_assertions)]
    fn label() -> &'static str {
        "Ray-Tracing Low shader"
    }
}
