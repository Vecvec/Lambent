use bytemuck::{Pod, Zeroable};
use core::panic;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::{Add, Range};
use wesl::{CompileOptions, Router, VirtualResolver, Wesl};
use wgpu::util::BufferInitDescriptor;
use wgpu::{
    BindGroupLayout, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    ShaderModuleDescriptor, ShaderSource,
};

pub mod camera;
pub mod data_buffer;
pub mod debug;
pub mod importance_sampling;
pub mod intersection_handlers;
pub mod low_level;
pub mod path_tracing;
#[cfg(test)]
mod tests;
pub mod textures;

use crate::low_level::rc_resolver::RcResolver;
use crate::low_level::{IntersectionHandler, RayTracingShader, RayTracingShaderDST};
pub use data_buffer::{AdvanceOptions, BufferType, DataBuffers};
pub use low_level::RayTracingOptions;

/// Refractive indices from https://refractiveindex.info/
#[allow(
    clippy::excessive_precision,
    reason = "These are the mesured values, so they should have as much precision as the real values"
)]
pub mod refractive_indices {
    use std::ops::Range;

    pub const fn refractive_index_between(in_front: Range<f32>, behind: Range<f32>) -> Range<f32> {
        (behind.start / in_front.start)..(behind.end / in_front.end)
    }

    /// Refractive index between a vacuum (in front) and a vacuum (behind).
    pub const VACUUM: Range<f32> = 1.0..1.0;
    /// Refractive index between a vacuum (in front) and air (behind).
    // https://refractiveindex.info/?shelf=other&book=air&page=Ciddor
    pub const AIR: Range<f32> = 1.00027531..1.000287;
    /// Refractive index between a vacuum (in front) and ice at -7°C (behind).
    // https://refractiveindex.info/?shelf=3d&book=crystals&page=ice
    pub const ICE: Range<f32> = 1.3057..1.3267;
    /// Refractive index between a vacuum (in front) and water at 25°C (behind).
    // https://refractiveindex.info/?shelf=3d&book=liquids&page=water
    pub const WATER: Range<f32> = 1.33..1.3442;
    /// Refractive index between a vacuum (in front) and fused quartz (behind).
    // https://refractiveindex.info/?shelf=glass&book=fused_silica&page=Malitson
    pub const GLASS: Range<f32> = 1.4540..1.4787;
    /// Refractive index between a vacuum (in front) and diamond (behind).
    // https://refractiveindex.info/?shelf=3d&book=crystals&page=diamond
    pub const DIAMOND: Range<f32> = 2.4063..2.4922;
}

pub struct Shader {
    pub(crate) base: String,
    #[cfg(debug_assertions)]
    pub(crate) label: &'static str,
}

pub trait Descriptor {
    /// Creates a buffer init descriptor from the type
    ///
    /// Returned buffer usage is for correctly binding it to the ray tracing shaders, plus `further_usages`
    fn buffer_descriptor(&self, further_usages: BufferUsages) -> BufferInitDescriptor<'_>;
}

impl Shader {
    pub fn descriptor(&self) -> ShaderModuleDescriptor<'_> {
        ShaderModuleDescriptor {
            #[cfg(debug_assertions)]
            label: Some(self.label),
            #[cfg(not(debug_assertions))]
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(&self.base)),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq, Eq, Hash)]
pub struct TexturePositions {
    tex_pos_base_colour_1: u32,
    tex_pos_base_colour_2: u32,
    tex_pos_base_colour_3: u32,
    tex_pos_emission_1: u32,
    tex_pos_emission_2: u32,
    tex_pos_emission_3: u32,
    tex_pos_attributes_1: u32,
    tex_pos_attributes_2: u32,
    tex_pos_attributes_3: u32,
    tex_pos_coat_1: u32,
    tex_pos_coat_2: u32,
    tex_pos_coat_3: u32,
    tex_pos_sub_surface_1: u32,
    tex_pos_sub_surface_2: u32,
    tex_pos_sub_surface_3: u32,
    tex_pos_metalness_1: u32,
    tex_pos_metalness_2: u32,
    tex_pos_metalness_3: u32,
    tex_pos_specular_1: u32,
    tex_pos_specular_2: u32,
    tex_pos_specular_3: u32,
}

impl TexturePositions {
    /// `tex_pos_specular` is for both the specular rougness and specular colour textures
    pub fn new(
        tex_pos_base_colour: [[f32; 2]; 3],
        tex_pos_emission: [[f32; 2]; 3],
        tex_pos_attributes: [[f32; 2]; 3],
        tex_pos_coat_colour: [[f32; 2]; 3],
        tex_pos_subsurface_colour: [[f32; 2]; 3],
        tex_pos_metalness: [[f32; 2]; 3],
        tex_pos_specular: [[f32; 2]; 3],
    ) -> Self {
        Self {
            tex_pos_base_colour_1: pack2xf16(tex_pos_base_colour[0]),
            tex_pos_base_colour_2: pack2xf16(tex_pos_base_colour[1]),
            tex_pos_base_colour_3: pack2xf16(tex_pos_base_colour[2]),
            tex_pos_emission_1: pack2xf16(tex_pos_emission[0]),
            tex_pos_emission_2: pack2xf16(tex_pos_emission[1]),
            tex_pos_emission_3: pack2xf16(tex_pos_emission[2]),
            tex_pos_attributes_1: pack2xf16(tex_pos_attributes[0]),
            tex_pos_attributes_2: pack2xf16(tex_pos_attributes[1]),
            tex_pos_attributes_3: pack2xf16(tex_pos_attributes[2]),
            tex_pos_coat_1: pack2xf16(tex_pos_coat_colour[0]),
            tex_pos_coat_2: pack2xf16(tex_pos_coat_colour[1]),
            tex_pos_coat_3: pack2xf16(tex_pos_coat_colour[2]),
            tex_pos_sub_surface_1: pack2xf16(tex_pos_subsurface_colour[0]),
            tex_pos_sub_surface_2: pack2xf16(tex_pos_subsurface_colour[1]),
            tex_pos_sub_surface_3: pack2xf16(tex_pos_subsurface_colour[2]),
            tex_pos_metalness_1: pack2xf16(tex_pos_metalness[0]),
            tex_pos_metalness_2: pack2xf16(tex_pos_metalness[1]),
            tex_pos_metalness_3: pack2xf16(tex_pos_metalness[2]),
            tex_pos_specular_1: pack2xf16(tex_pos_specular[0]),
            tex_pos_specular_2: pack2xf16(tex_pos_specular[1]),
            tex_pos_specular_3: pack2xf16(tex_pos_specular[2]),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default, PartialEq, Eq, Hash)]
pub struct Material {
    tex_pos_recolour: u32,
    tex_idx_base_colour_emission: u32,
    tex_idx_roughness_coat: u32,
    tex_idx_subsurface_metalness: u32,
    tex_idx_specular_colour_specular_rougness: u32,
    /// A combination of the following:
    /// - The material type (8 bit unsigned int)
    /// - Whether the geometry is thin walled (1 bit)
    /// - The subsurface scattering wieght of the object (7 bit unsigned int)
    /// - The subsurface anisotopy  (16 bit signed int)
    ty_thin_walled_subsurface: u32,
    emission_scale: u32,
    refractive_index: u32,
    specular_refractive_index: u32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct EmissionScale {
    /// The emission scale for the
    pub emissive_texture_scale: half::bf16,
    pub diffuse_texture_scale: half::bf16,
}

impl EmissionScale {
    pub fn from_emission_scale(scale: f32) -> Self {
        Self {
            emissive_texture_scale: half::bf16::from_f32(scale),
            diffuse_texture_scale: half::bf16::ZERO,
        }
    }
    pub fn from_f32(emissive_texture_scale: f32, diffuse_texture_scale: f32) -> Self {
        Self {
            emissive_texture_scale: half::bf16::from_f32(emissive_texture_scale),
            diffuse_texture_scale: half::bf16::from_f32(diffuse_texture_scale),
        }
    }
    fn pack(self) -> u32 {
        let float_1 = self.emissive_texture_scale.to_ne_bytes();
        let float_2 = self.diffuse_texture_scale.to_ne_bytes();
        <u32>::from_ne_bytes([float_1[0], float_1[1], float_2[0], float_2[1]])
    }
}

impl Material {
    /// texture index (the index into the texture loader),
    /// the brightness scale of the emission scale, the
    /// refractive index, and the type.
    ///
    /// - refractive_index: and optional range between
    ///   refractive index at 760 nm (red) and 340 nm (violet)
    ///
    /// Note: Not all these types will result in a different
    /// Material. Things such as small refractive indices
    /// are rounded to the nearest bf16
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tex_pos_recolour: [u16; 2],
        tex_idx_base_colour: u16,
        tex_idx_emission: Option<u16>,
        tex_idx_roughness: Option<u16>,
        tex_idx_coat: Option<u16>,
        tex_idx_subsurface: Option<u16>,
        tex_idx_metalness: Option<u16>,
        tex_idx_specular_colour: Option<u16>,
        tex_idx_specular_roughness: Option<u16>,
        emission_scale: EmissionScale,
        coat_refractive_index: Option<Range<f32>>,
        specular_refractive_index: Option<Range<f32>>,
        ty: MaterialType,
        thin_walled: bool,
        subsurface_weight: f32,
        subsurface_anisotropy: f32,
    ) -> Self {
        let tex_pos_recolour = pack2xu16(tex_pos_recolour);
        let idx = coat_refractive_index.unwrap_or(refractive_indices::VACUUM);
        let specular_idx = specular_refractive_index.unwrap_or(refractive_indices::VACUUM);

        let subsurface_weight = (subsurface_weight.clamp(0.0, 1.0) * ((1 << 7) - 1) as f32) as u8;
        let subsurface_anisotropy =
            ((subsurface_anisotropy.clamp(-1.0, 1.0) * (i16::MAX as f32)) as i16).to_be_bytes();

        Self {
            tex_pos_recolour,
            tex_idx_base_colour_emission: pack2xu16([
                tex_idx_base_colour,
                map_optional_idx(tex_idx_emission),
            ]),
            tex_idx_roughness_coat: pack2xu16([
                map_optional_idx(tex_idx_roughness),
                map_optional_idx(tex_idx_coat),
            ]),
            tex_idx_subsurface_metalness: pack2xu16([
                map_optional_idx(tex_idx_subsurface),
                map_optional_idx(tex_idx_metalness),
            ]),
            tex_idx_specular_colour_specular_rougness: pack2xu16([
                map_optional_idx(tex_idx_specular_colour),
                map_optional_idx(tex_idx_specular_roughness),
            ]),
            ty_thin_walled_subsurface: u32::from_ne_bytes([
                ty as _,
                subsurface_weight | (thin_walled as u8) << 7,
                subsurface_anisotropy[0],
                subsurface_anisotropy[1],
            ]),
            emission_scale: emission_scale.pack(),
            refractive_index: pack2xf16([idx.start, idx.end]),
            specular_refractive_index: pack2xf16([specular_idx.start, specular_idx.end]),
        }
    }
}

fn map_optional_idx(op: Option<u16>) -> u16 {
    op.map_or(<u16>::MAX, |u| {
        assert_ne!(<u16>::MAX, u);
        u
    })
}

fn pack2xf16(floats: [f32; 2]) -> u32 {
    let float_1 = half::f16::from_f32(floats[0]).to_ne_bytes();
    let float_2 = half::f16::from_f32(floats[1]).to_ne_bytes();
    <u32>::from_ne_bytes([float_1[0], float_1[1], float_2[0], float_2[1]])
}

fn pack2xu16(int: [u16; 2]) -> u32 {
    let int_1 = int[0].to_ne_bytes();
    let int_2 = int[1].to_ne_bytes();
    <u32>::from_ne_bytes([int_1[0], int_1[1], int_2[0], int_2[1]])
}

impl Descriptor for [Material] {
    fn buffer_descriptor(&self, further_usages: BufferUsages) -> BufferInitDescriptor<'_> {
        BufferInitDescriptor {
            label: Some("Materials"),
            contents: bytemuck::cast_slice(self),
            usage: BufferUsages::STORAGE | further_usages,
        }
    }
}

impl Descriptor for [TexturePositions] {
    fn buffer_descriptor(&self, further_usages: BufferUsages) -> BufferInitDescriptor<'_> {
        BufferInitDescriptor {
            label: Some("Materials"),
            contents: bytemuck::cast_slice(self),
            usage: BufferUsages::STORAGE | further_usages,
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
pub enum MaterialType {
    /// A material the randomly bounces in a hemisphere
    Diffuse = 0,
    /// A material that always reflects
    ///
    /// Equivalent to diffuse with metalness forced to 1.
    Metallic = 1,
    /// A material that light passes through (mostly, some reflects)
    Transparent = 2,
}

pub type MaterialIndices = Vec<u32>;

/// the size for a dispatch
pub struct DispatchSize {
    pub width: u32,
    pub height: u32,
}

/// calculate the size for a dispatch of the ray-tracing compute function
pub fn dispatch_size(width: u32, height: u32) -> DispatchSize {
    DispatchSize {
        width: width.div_ceil(64),
        height: height.div_ceil(1),
    }
}

pub struct DynamicRayTracer {
    device: Device,
    intersection_handler: String,
    shader: Box<dyn RayTracingShaderDST>,
    extra_bgls: Vec<BindGroupLayout>,
    resolver: Option<RcResolver>,
}

impl DynamicRayTracer {
    pub fn set_intersection_handler(&mut self, handler: &dyn IntersectionHandler) {
        self.intersection_handler = handler.source();
        self.extra_bgls = handler.additional_bind_group_layouts(&self.device);
    }

    pub fn create_pipeline<Opts: low_level::RayTracerOptions>(
        &self,
        blas_count: NonZeroU32,
        texture_count: NonZeroU32,
        options: &Opts,
    ) -> ComputePipeline {
        create_pipeline(
            blas_count,
            texture_count,
            options,
            self.shader
                .shader_source_without_intersection_handler(options),
            &self.device,
            &self.extra_bgls,
            &self.intersection_handler,
            None,
            &self.resolver,
        )
    }
}

pub struct RayTracer<S: RayTracingShader> {
    device: Device,
    intersection_handler: String,
    shader: PhantomData<S>,
    extra_bgls: Vec<BindGroupLayout>,
    resolver: Option<RcResolver>,
}

impl<S: RayTracingShader> RayTracer<S> {
    pub fn new(device: &Device) -> Self {
        let mut tracer = Self {
            device: device.clone(),
            intersection_handler: "".to_string(),
            shader: PhantomData,
            extra_bgls: Vec::new(),
            resolver: None,
        };
        tracer.set_intersection_handler(&intersection_handlers::DefaultIntersectionHandler);
        tracer
    }

    pub fn required_features() -> wgpu::Features {
        S::features()
    }

    pub fn required_limits() -> wgpu::Limits {
        S::limits().or_better_values_from(
            &wgpu::Limits::default().using_minimum_supported_acceleration_structure_values(),
        )
    }

    pub fn set_intersection_handler(&mut self, handler: &dyn IntersectionHandler) {
        self.intersection_handler = handler.source();
        self.extra_bgls = handler.additional_bind_group_layouts(&self.device);
        self.resolver = handler.resolver().map(RcResolver);
    }

    pub fn create_pipeline<Opts: low_level::RayTracerOptions>(
        &self,
        blas_count: NonZeroU32,
        texture_count: NonZeroU32,
        options: &Opts,
    ) -> ComputePipeline {
        create_pipeline(
            blas_count,
            texture_count,
            options,
            S::shader_source_without_intersection_handler(options),
            &self.device,
            &self.extra_bgls,
            &self.intersection_handler,
            get_label::<S>(),
            &self.resolver,
        )
    }

    pub fn dynamic(self) -> DynamicRayTracer {
        DynamicRayTracer {
            device: self.device,
            intersection_handler: self.intersection_handler,
            shader: Box::new(S::new()),
            extra_bgls: self.extra_bgls,
            resolver: self.resolver,
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn create_pipeline<Opts: low_level::RayTracerOptions>(
    blas_count: NonZeroU32,
    texture_count: NonZeroU32,
    options: &Opts,
    src: String,
    device: &Device,
    extra_bgls: &[BindGroupLayout],
    intersection_handler: &str,
    label: Option<&str>,
    resolver: &Option<RcResolver>,
) -> ComputePipeline {
    let pipeline_layout = low_level::pipeline_layout(device, blas_count, texture_count, extra_bgls);

    let compiled = compile_shader(src, intersection_handler, resolver);

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label,
        source: ShaderSource::Wgsl(Cow::Owned(compiled)),
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("rt_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[
                ("SAMPLES", options.samples().get() as f64),
                ("T_MIN", options.t_min() as f64),
                ("T_MAX", options.t_max() as f64),
            ],
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    })
}

fn compile_shader(
    module: String,
    intersection_handler: &str,
    other_resolver: &Option<RcResolver>,
) -> String {
    // Compile the partial module with the intersection handler.
    let mut compiler = Wesl::new_barebones();
    compiler.set_options(CompileOptions {
        validate: false,
        ..Default::default()
    });

    let path_tracer_module = "import package::intersection_handler::intersect;\n\n"
        .to_string()
        .add(&module);

    let mut resolver = VirtualResolver::new();
    resolver.add_module(
        "package::path_tracer".parse().unwrap(),
        std::borrow::Cow::Owned(path_tracer_module),
    );

    resolver.add_module(
        "package::intersection_handler".parse().unwrap(),
        std::borrow::Cow::Owned(intersection_handler.to_string()),
    );

    resolver.add_module(
        "package::intersection".parse().unwrap(),
        std::borrow::Cow::Borrowed(include_str!("intersection.wesl")),
    );

    let mut router = Router::new();
    if let Some(resolver) = other_resolver.clone() {
        router.mount_fallback_resolver(resolver);
    }
    router.mount_resolver("package".parse().unwrap(), resolver);
    let mut compiler = Wesl::new_barebones().set_custom_resolver(router);

    compiler.set_options(CompileOptions {
        validate: false,
        ..Default::default()
    });

    compiler
        .compile(&"package::path_tracer".parse().unwrap())
        .inspect_err(|e| panic!("{e}"))
        .unwrap()
        .to_string()
}

#[cfg(debug_assertions)]
fn get_label<S: RayTracingShader>() -> Option<&'static str> {
    Some(S::label())
}

#[cfg(not(debug_assertions))]
fn get_label<S: RayTracingShader>() -> Option<&'static str> {
    None
}

/// matches the verices structure added to bindings.wgsl if feature `no-vertex-return` is enabled
#[repr(C)]
#[derive(Debug)]
pub struct Vertices {
    pub geometry_stride: u32,
    pub vertices: Vec<[f32; 4]>,
}

impl Vertices {
    pub fn append_bytes(&self, bytes: &mut Vec<u8>) {
        bytes.reserve_exact(size_of_val(self) + size_of::<[f32; 3]>());
        bytes.extend_from_slice(&self.geometry_stride.to_ne_bytes());
        bytes.extend_from_slice(&[0; size_of::<[f32; 3]>()]);
        bytes.extend_from_slice(bytemuck::cast_slice(&self.vertices));
    }
}
