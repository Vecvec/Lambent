#[cfg(feature = "wip-features")]
use std::num::NonZeroU32;

#[cfg(feature = "wip-features")]
use crate::{Shader, intersection_handlers, low_level::IntersectionHandler};
#[cfg(feature = "wip-features")]
use wgpu::{BindGroupLayout, ComputePipeline, Device, Features};

#[cfg(feature = "wip-features")]
/// A shader for importance sampling. Based off reSTIR GI.
pub struct SpatialResampling {
    device: Device,
    extra_bgls: Vec<BindGroupLayout>,
}

#[cfg(feature = "wip-features")]
impl SpatialResampling {
    pub fn new(device: &Device) -> Self {
        let mut resampler = Self {
            device: device.clone(),
            // intersection_handler: "".to_string(),
            // shader: PhantomData,
            extra_bgls: Vec::new(),
            // resolver: None,
        };
        resampler.set_intersection_handler(&intersection_handlers::DefaultIntersectionHandler);
        resampler
    }

    pub fn set_intersection_handler(&mut self, handler: &dyn IntersectionHandler) {
        // self.intersection_handler = handler.source();
        self.extra_bgls = handler.additional_bind_group_layouts(&self.device);
        // self.resolver = handler.resolver().map(|h| RcResolver(h));
    }

    pub fn features() -> Features {
        // features required to interact
        Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    }
    pub fn create_shader() -> Shader {
        use wesl::include_wesl;

        Shader {
            base: include_wesl!("spatial_resampling").to_string(),
            #[cfg(debug_assertions)]
            label: "Spatial Resampler",
        }
    }

    pub fn create_pipeline(
        &self,
        blas_count: NonZeroU32,
        diffuse_count: NonZeroU32,
        emission_count: NonZeroU32,
        attribute_count: NonZeroU32,
        overrides: &[(&str, f64)],
    ) -> ComputePipeline {
        use wgpu::ComputePipelineDescriptor;

        use crate::low_level;

        let pipeline_layout = low_level::pipeline_layout(
            &self.device,
            blas_count,
            diffuse_count,
            emission_count,
            attribute_count,
            &self.extra_bgls,
        );

        let shader = self.device.create_shader_module(Self::create_shader().descriptor());

        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: overrides,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            })
    }
}

#[cfg(feature = "wip-features")]
/// A shader for importance sampling. Based off reSTIR GI.
pub struct TemporalResampling {
    device: Device,
    extra_bgls: Vec<BindGroupLayout>,
}

#[cfg(feature = "wip-features")]
impl TemporalResampling {
    pub fn new(device: &Device) -> Self {
        let mut resampler = Self {
            device: device.clone(),
            // intersection_handler: "".to_string(),
            // shader: PhantomData,
            extra_bgls: Vec::new(),
            // resolver: None,
        };
        resampler.set_intersection_handler(&intersection_handlers::DefaultIntersectionHandler);
        resampler
    }

    pub fn set_intersection_handler(&mut self, handler: &dyn IntersectionHandler) {
        // self.intersection_handler = handler.source();
        self.extra_bgls = handler.additional_bind_group_layouts(&self.device);
        // self.resolver = handler.resolver().map(|h| RcResolver(h));
    }

    pub fn features() -> Features {
        // features required to interact
        Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
    }
    fn create_shader() -> Shader {
        use wesl::include_wesl;

        Shader {
            base: include_wesl!("temporal_resampling").to_string(),
            #[cfg(debug_assertions)]
            label: "Temporal Resampler",
        }
    }

    pub fn create_pipeline(
        &self,
        blas_count: NonZeroU32,
        diffuse_count: NonZeroU32,
        emission_count: NonZeroU32,
        attribute_count: NonZeroU32,
    ) -> ComputePipeline {
        use wgpu::ComputePipelineDescriptor;

        use crate::low_level;

        let pipeline_layout = low_level::pipeline_layout(
            &self.device,
            blas_count,
            diffuse_count,
            emission_count,
            attribute_count,
            &self.extra_bgls,
        );

        let shader = self.device.create_shader_module(Self::create_shader().descriptor());

        self.device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

pub(crate) struct Reservoir {
    _visible_point: [f32; 3],
    _visible_normal: u32,
    _sample_point: [f32; 3],
    _sample_normal: u32,
    _out_radiance: [f32; 3],
    _ty: u32,
    _roughness: f32,
    _pdf: f32,
    _w: f32,
    _m_valid: u32,
    _full_w: f32,
    _pad: [f32; 3],
}

pub(crate) struct Info {
    _emission: [f32; 4],
    _albedo: [f32; 4],
    _cam_loc: [f32; 4],
}
