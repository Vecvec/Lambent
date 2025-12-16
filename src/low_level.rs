use crate::{DynamicRayTracer, RayTracer};
use std::{num::NonZeroU32, rc::Rc};
use wesl::Resolver;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, BufferSize, Device, Features, Limits, PipelineLayout,
    PipelineLayoutDescriptor, SamplerBindingType, ShaderStages,
    StorageTextureAccess, TextureFormat, TextureSampleType, TextureViewDimension,
};

pub(crate) mod rc_resolver {
    use std::{borrow::Cow, path::PathBuf, rc::Rc};

    use wesl::{syntax::TranslationUnit, ModulePath, ResolveError, Resolver};

    #[derive(Clone)]
    pub(crate) struct RcResolver(pub(crate) Rc<dyn Resolver>);

    impl Resolver for RcResolver {
        fn resolve_source<'a>(&'a self, path: &ModulePath) -> Result<Cow<'a, str>, ResolveError> {
            (*self.0).resolve_source(path)
        }
        fn resolve_module(&self, path: &ModulePath) -> Result<TranslationUnit, ResolveError> {
            (*self.0).resolve_module(path)
        }
        fn display_name(&self, path: &ModulePath) -> Option<String> {
            (*self.0).display_name(path)
        }
        fn fs_path(&self, path: &ModulePath) -> Option<PathBuf> {
            (*self.0).fs_path(path)
        }
    }
}

/// Source in WESL. Must define a function with the signature `fn intersect(intersection: RayIntersection) -> AABBIntersection`
///
/// Note: `AABBIntersection` is defined as
/// ````wgsl
/// struct AABBIntersection {
///     hit: bool,
///     normal: vec3<f32>,
///     t: f32,
/// }
/// ````
///
/// It is considered a logic error if the module does not pass validation (but should *not* cause UB)
///
/// # Defined Packages
///
/// `package::intersection` defines the AABBIntersection.
/// `package::intersection_handler` is the module returned by `source`.
/// `package::path_tracer` defines the main path tracer, and should *not* be used, its contents changing is not considered breaking.
///
/// You may specify a fallback resolver that will resolve after these defined packages are also called.
///
/// # Safety:
///
/// - The function returned by `source`, when executed, *must* return in a finite amount of time.
pub unsafe trait IntersectionHandler: 'static {
    fn source(&self) -> String;
    fn resolver(&self) -> Option<Rc<dyn Resolver>> {
        None
    }
    fn additional_bind_group_layouts(&self, _device: &Device) -> Vec<BindGroupLayout> {
        Vec::new()
    }
}

/// It is considered a logic error if the module does not pass validation (but does *not* cause UB)
///
/// # Safety:
///
/// - The shader returned by `create_shader`, when executed, *must* return in a finite amount of time.
pub unsafe trait RayTracingShader: Sized + 'static {
    fn new() -> Self;
    fn features() -> Features {
        #[cfg(no_vertex_return)]
        let maybe_vertex_return = Features::empty();
        #[cfg(not(no_vertex_return))]
        let maybe_vertex_return = Features::EXPERIMENTAL_RAY_HIT_VERTEX_RETURN;
        // features required to interact
        Features::EXPERIMENTAL_RAY_QUERY
            | Features::STORAGE_RESOURCE_BINDING_ARRAY
            | Features::BUFFER_BINDING_ARRAY
            | Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::IMMEDIATES
            | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | Features::TEXTURE_BINDING_ARRAY
            | Features::PARTIALLY_BOUND_BINDING_ARRAY
            | Features::SUBGROUP
            | maybe_vertex_return
    }
    fn limits() -> Limits {
        // limits required to interact
        Limits {
            max_immediate_size: 4,
            max_storage_buffer_binding_size: Limits::default().max_storage_buffer_binding_size,
            // see docs for why 500,000.
            max_binding_array_elements_per_shader_stage: 500_000,
            ..Limits::default()
        }
    }
    #[deprecated]
    fn limits_or(limit: Limits) -> Limits {
        // limits required to interact
        Limits {
            max_immediate_size: 4.max(limit.max_immediate_size),
            max_storage_buffer_binding_size: (Limits::default().max_storage_buffer_binding_size)
                .max(limit.max_storage_buffer_binding_size),
            max_binding_array_elements_per_shader_stage: 500_000
                .max(limit.max_binding_array_elements_per_shader_stage),
            max_acceleration_structures_per_shader_stage: 16
                .max(limit.max_acceleration_structures_per_shader_stage),
            max_blas_geometry_count: ((1 << 24) - 1).max(limit.max_blas_geometry_count),
            max_tlas_instance_count: ((1 << 24) - 1).max(limit.max_tlas_instance_count),
            max_blas_primitive_count: (1 << 28).max(limit.max_blas_primitive_count),
            ..limit
        }
    }
    fn shader_source_without_intersection_handler() -> String;
    #[cfg(debug_assertions)]
    fn label() -> &'static str;
}

/// Exactly the same as the ray-tracing trait but takes itself so can be made into a dst
///
/// Implementors should implement [RayTracingShader] instead
///
/// # Safety:
///
/// Same as [RayTracingShader]
pub unsafe trait RayTracingShaderDST {
    fn features(&self) -> Features;
    fn limits(&self) -> Limits;
    #[deprecated]
    fn limits_or(&self, limit: Limits) -> Limits;
    fn shader_source_without_intersection_handler(&self) -> String;
    #[cfg(debug_assertions)]
    fn label(&self) -> &'static str;
    fn dyn_ray_tracer(&self, device: &Device) -> DynamicRayTracer;
}

// # Safety:
//
// The safety requirements of `RayTracingShader` and `RayTracingShaderDST` are the same so anyone
// who has implemented `RayTracingShader` must have met the guarantees of `RayTracingShaderDST`.
unsafe impl<T: RayTracingShader> RayTracingShaderDST for T {
    fn features(&self) -> Features {
        T::features()
    }
    fn limits(&self) -> Limits {
        T::limits()
    }
    // this method is also depricated
    #[expect(deprecated)]
    fn limits_or(&self, limit: Limits) -> Limits {
        T::limits_or(limit)
    }
    fn shader_source_without_intersection_handler(&self) -> String {
        T::shader_source_without_intersection_handler()
    }
    #[cfg(debug_assertions)]
    fn label(&self) -> &'static str {
        T::label()
    }
    fn dyn_ray_tracer(&self, device: &Device) -> DynamicRayTracer {
        RayTracer::<T>::new(device).dynamic()
    }
}

pub fn pipeline_layout(
    device: &Device,
    blas_count: NonZeroU32,
    diffuse_count: NonZeroU32,
    emission_count: NonZeroU32,
    attribute_count: NonZeroU32,
    extra_bgls: &[BindGroupLayout],
) -> PipelineLayout {
    #[cfg(not(no_vertex_return))]
    let entries = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(44).unwrap()),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(4).unwrap()),
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::AccelerationStructure {
                vertex_return: true,
            },
            count: None,
        },
    ];
    #[cfg(no_vertex_return)]
    let entries = &[
        BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(32).unwrap()),
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(BufferSize::new(4).unwrap()),
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 2,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::AccelerationStructure {
                vertex_return: false,
            },
            count: None,
        },
        BindGroupLayoutEntry {
            binding: 3,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: Some(blas_count),
        },
        BindGroupLayoutEntry {
            binding: 4,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: Some(blas_count),
        },
    ];
    let mat_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries,
    });
    let mut bgls = Vec::with_capacity(extra_bgls.len() + 3);
    bgls.push(&mat_bgl);
    let out_bgl = out_bgl(device);
    bgls.push(&out_bgl);
    let texture_bgl = texture_bgl(device, [diffuse_count, emission_count, attribute_count]);
    bgls.push(&texture_bgl);
    bgls.extend(extra_bgls.iter());
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&mat_bgl, &out_bgl, &texture_bgl],
        immediate_size: 4,
    })
}

pub(crate) fn texture_bgl(device: &Device, counts: [NonZeroU32; 3]) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[0]),
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[1]),
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(counts[2]),
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

pub(crate) fn out_bgl(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: Some(BufferSize::new(128).unwrap()),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: TextureFormat::Rgba32Float,
                    view_dimension: TextureViewDimension::D2,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::Cube,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 10,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 11,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}
