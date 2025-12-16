use cgmath::{Matrix4, Point3, Vector3};
use futures::executor::block_on;
use glfw::{fail_on_errors, ClientApiHint, WindowHint, WindowMode};
use lambent::camera::Camera;
use lambent::importance_sampling::SpatialResampling;
use lambent::textures::TextureLoader;
use lambent::{
    dispatch_size, low_level::pipeline_layout, path_tracing, textures, DataBuffers, Descriptor,
    Material, MaterialType,
};
use lambent::{refractive_indices, BufferType};
use std::cmp::{max, min};
use std::marker::PhantomData;
use std::num::NonZeroU32;
#[cfg(feature = "denoise")]
use std::sync::mpsc;
#[cfg(feature = "denoise")]
use std::time::Duration;
use std::time::Instant;
use std::{iter, mem};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
#[cfg(no_vertex_return)]
use wgpu::BufferBinding;
#[cfg(feature = "denoise")]
use wgpu::MapMode;
use wgpu::{
    include_wgsl, AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, BlasBuildEntry,
    BlasGeometries, BlasGeometrySizeDescriptors, BlasTriangleGeometry,
    BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferBindingType, BufferUsages,
    ColorTargetState, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipelineDescriptor,
    CreateBlasDescriptor, CreateTlasDescriptor, DeviceDescriptor, ExperimentalFeatures, Extent3d,
    Features, FragmentState, IndexFormat, InstanceDescriptor, Limits, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PresentMode,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipelineDescriptor,
    RequestAdapterOptions, ShaderStages, SurfaceError, TextureDescriptor, TextureDimension,
    TextureFormat, TextureSampleType, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    TlasInstance, VertexFormat, VertexState,
};

type RayTracer = lambent::RayTracer<path_tracing::Medium>;

const SIZE: u32 = 320;

const SAMPLES: usize = 1;

const IS_SAMPLES: usize = 4;

const IS_SPACE: usize = 31;

const LIGHT_SIZE: f32 = 0.1;
/// Total light produced, correlated to lumens.
const LIGHT_BRIGHTNESS: f32 = 1.0;
/// The brightness of the light at a point.
const POINT_BRIGHTNESS: f32 = LIGHT_BRIGHTNESS / (LIGHT_SIZE * LIGHT_SIZE);

fn main() {
    env_logger::init();
    let mut samples = SAMPLES;
    let args = std::env::args();

    let mut change_seed = true;
    let mut importance_sampling = false;
    let mut change_light_brightness = false;

    let mut maximum = 256;
    for arg in args {
        if arg == "no-change-seed" {
            change_seed = false;
            maximum = 1;
        }
        if arg == "no-average" {
            maximum = 1;
        }
        if arg == "importance-sampling" {
            importance_sampling = true;
        }
        if arg == "change-light-brightness" {
            change_light_brightness = true;
        }
    }

    for var in std::env::vars() {
        if var.0 == "samples" {
            let override_samples = var.1.parse::<usize>();
            match override_samples {
                Ok(sam) => {
                    samples = sam;
                }
                Err(err) => {
                    println!("error parsing samples: {err}");
                }
            }
        }
    }

    // For a real scene, these should be generated
    let positions = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let materials = &[
        // since these textures are a solid color it is only required for one of each material
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            0,
            None,
            Some(0),
            1.0,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            1,
            None,
            None,
            1.0,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            2,
            None,
            None,
            1.0,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            3,
            Some(0),
            None,
            POINT_BRIGHTNESS,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            0,
            None,
            None,
            1.0,
            Some(refractive_indices::refractive_index_between(
                refractive_indices::AIR,
                refractive_indices::DIAMOND,
            )),
            MaterialType::Transparent,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0; 2],
            0,
            None,
            None,
            1.0,
            None,
            MaterialType::Metallic,
        ),
    ];

    let light_pos = [0.5 - (LIGHT_SIZE / 2.0), 0.5 + (LIGHT_SIZE / 2.0)];
    const SMALL_BLOCK_OFFSET: f32 = 0.15;
    const LIGHT_HEIGHT: f32 = 0.999999;
    let vertices = lambent::Vertices {
        geometry_stride: 0,
        vertices: vec![
            [0.0f32, 0.0, 0.0, 0.0],                         //0
            [1.0, 0.0, 0.0, 0.0],                            //1
            [0.0, 0.0, 1.0, 0.0],                            //2
            [1.0, 0.0, 1.0, 0.0],                            //3
            [0.0, 1.0, 0.0, 0.0],                            //4
            [1.0, 1.0, 0.0, 0.0],                            //5
            [0.0, 1.0, 1.0, 0.0],                            //6
            [1.0, 1.0, 1.0, 0.0],                            //7
            [light_pos[0], LIGHT_HEIGHT, light_pos[0], 0.0], //8
            [light_pos[1], LIGHT_HEIGHT, light_pos[0], 0.0], //9
            [light_pos[0], LIGHT_HEIGHT, light_pos[1], 0.0], //10
            [light_pos[1], LIGHT_HEIGHT, light_pos[1], 0.0], //11
            [0.62, 0.0, 0.44, 0.0],
            [0.44169, 0.0, 0.61523, 0.0],
            [0.61691, 0.0, 0.79354, 0.0],
            [0.79523, 0.0, 0.61831, 0.0],
            [0.62, 0.66, 0.44, 0.0],
            [0.44169, 0.66, 0.61523, 0.0],
            [0.61691, 0.66, 0.79354, 0.0],
            [0.79523, 0.66, 0.61831, 0.0],
            [SMALL_BLOCK_OFFSET, 0.0, SMALL_BLOCK_OFFSET, 0.0],
            [SMALL_BLOCK_OFFSET + 0.3, 0.0, SMALL_BLOCK_OFFSET, 0.0],
            [SMALL_BLOCK_OFFSET + 0.3, 0.0, SMALL_BLOCK_OFFSET + 0.3, 0.0],
            [SMALL_BLOCK_OFFSET, 0.0, SMALL_BLOCK_OFFSET + 0.3, 0.0],
            [SMALL_BLOCK_OFFSET, 0.3, SMALL_BLOCK_OFFSET, 0.0],
            [SMALL_BLOCK_OFFSET + 0.3, 0.3, SMALL_BLOCK_OFFSET, 0.0],
            [SMALL_BLOCK_OFFSET + 0.3, 0.3, SMALL_BLOCK_OFFSET + 0.3, 0.0],
            [SMALL_BLOCK_OFFSET, 0.3, SMALL_BLOCK_OFFSET + 0.3, 0.0],
        ],
    };
    #[rustfmt::skip]
    let indices = [
        0u32, 1, 2, 1, 2, 3,
        0, 4, 2, 4, 2, 6,
        4, 5, 6, 5, 6, 7,
        2, 3, 6, 3, 6, 7,
        1, 3, 5, 3, 5, 7,
        8, 9, 10, 9, 10, 11,
        16, 17, 18, 16, 18, 19,
        17, 16, 12, 17, 12, 13,
        18, 17, 14, 13, 14, 17,
        19, 18, 15, 14, 15, 18,
        16, 19, 15, 16, 15, 12,
        // Small cube
        24, 26, 25, 24, 27, 26,
        // front
        25, 20, 24, 25, 21, 20,
        // left side (from front)
        26, 22, 25, 21, 25, 22,
        // back
        27, 23, 26, 22, 26, 23,
        // right side (from front)
        24, 23, 27, 24, 20, 23,
    ];
    let mat_indices = [
        0u32, 0, 2, 2, 0, 0, 0, 0, 1, 1, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4,
    ];

    let instance = wgpu::Instance::new(&InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&RequestAdapterOptions::default()))
        .expect("failed to find a suitable adapter");
    // a reasonable limit (128 if the seed changes)
    let target_exe_num = min(
        adapter.limits().max_binding_array_elements_per_shader_stage as usize,
        maximum,
    );
    let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
        label: None,
        required_features: RayTracer::required_features()
            | Features::TEXTURE_BINDING_ARRAY
            | Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            | Features::BGRA8UNORM_STORAGE
            | SpatialResampling::features(),
        // it's recommended to only use limits you need (due to possible perf issues)
        required_limits: RayTracer::required_limits().or_better_values_from(&Limits {
            max_binding_array_elements_per_shader_stage: max(
                target_exe_num as u32,
                Limits::default().max_binding_array_elements_per_shader_stage,
            ),
            ..Limits::default()
        }).using_minimum_supported_acceleration_structure_values(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: Default::default(),
        experimental_features: unsafe { ExperimentalFeatures::enabled() },
    }))
    .unwrap();
    println!(
        "targeting {} samples with {} execution(s)",
        samples * target_exe_num,
        target_exe_num
    );

    let ray_tracer = RayTracer::new(&device);

    let mut glfw = glfw::init(fail_on_errors!()).unwrap();
    // on some platforms this fixes crashes
    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));

    glfw.window_hint(WindowHint::Resizable(false));
    let (mut window, _) = glfw
        .create_window(SIZE, SIZE, "Cornell Box", WindowMode::Windowed)
        .unwrap();

    let surface = instance.create_surface(window.render_context()).unwrap();

    let mut surface_config = surface.get_default_config(&adapter, SIZE, SIZE).unwrap();
    surface_config.format = TextureFormat::Bgra8Unorm;
    surface_config.usage |= TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    surface_config.present_mode = PresentMode::AutoNoVsync;
    surface.configure(&device, &surface_config);

    let mut loader_diffuse = TextureLoader::new();
    loader_diffuse
        .load_from_bytes(include_bytes!("light_gray.png"))
        .unwrap();
    loader_diffuse
        .load_from_bytes(include_bytes!("green.png"))
        .unwrap();
    loader_diffuse
        .load_from_bytes(include_bytes!("red.png"))
        .unwrap();
    loader_diffuse
        .load_from_bytes(include_bytes!("pure_white.png"))
        .unwrap();
    let mut loader_emission = TextureLoader::new();
    loader_emission
        .load_from_bytes(include_bytes!("pure_white.png"))
        .unwrap();
    let mut loader_attributes = TextureLoader::new();
    loader_attributes
        .load_from_bytes(include_bytes!("pure_red.png"))
        .unwrap();
    let diffuse_textures = loader_diffuse.create_textures(&device, &queue, TextureUsages::empty());
    let emission_textures =
        loader_emission.create_textures(&device, &queue, TextureUsages::empty());
    let attribute_textures =
        loader_attributes.create_textures(&device, &queue, TextureUsages::empty());

    let buffers = DataBuffers::new(&device, SIZE, SIZE, 1_500_000, BufferType::all());
    let (texture_bg, _) = textures::bind_group_from_textures(
        &device,
        &queue,
        &diffuse_textures,
        &emission_textures,
        &attribute_textures,
        None,
        None,
    );

    let pipeline_layout = pipeline_layout(
        &device,
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(4).unwrap(),
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(1).unwrap(),
        &[],
    );

    let compute_pipeline = ray_tracer.create_pipeline(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(4).unwrap(),
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(1).unwrap(),
        &[("SAMPLES", SAMPLES as f64)],
    );

    let is_shader = SpatialResampling::create_shader();
    let is_shader = device.create_shader_module(is_shader.descriptor());
    let is_compute_pipeline: wgpu::ComputePipeline =
        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &is_shader,
            entry_point: None,
            compilation_options: PipelineCompilationOptions {
                constants: &[
                    //("SAMPLES", SAMPLES as f64),
                    ("IS_SAMPLES", IS_SAMPLES as f64),
                    ("IS_SPACE", IS_SPACE as f64),
                    //("IMPORTANCE_LIKELIHOOD".to_string(), 0.3),
                ],
                ..Default::default()
            },
            cache: None,
        });

    #[cfg(feature = "denoise")]
    let oidn_device = oidn::Device::new();
    #[cfg(feature = "denoise")]
    let mut oidn_state = OidnState::new(&device, &oidn_device);
    #[cfg(not(feature = "denoise"))]
    let oidn_state = OidnState::new(&device);

    let averaging_pipeline_in_layout =
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: Some(NonZeroU32::new(target_exe_num as u32).unwrap()),
            }],
        });

    let averaging_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&averaging_pipeline_in_layout, &oidn_state.staging_bgl],
        immediate_size: 4,
    });

    let average_shader = device.create_shader_module(include_wgsl!("average_to_buf.wgsl"));

    let average_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: None,
        layout: Some(&averaging_pipeline_layout),
        module: &average_shader,
        entry_point: Some("average"),
        compilation_options: PipelineCompilationOptions {
            constants: &[("SIZE", SIZE as f64)],
            ..Default::default()
        },
        cache: None,
    });

    let copy_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&oidn_state.staging_bgl],
        immediate_size: 0,
    });

    let copy_shader = device.create_shader_module(include_wgsl!("buf_rgb_to_tex_rgba.wgsl"));

    let copy_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
        label: None,
        layout: Some(&copy_pipeline_layout),
        vertex: VertexState {
            module: &copy_shader,
            entry_point: None,
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(FragmentState {
            module: &copy_shader,
            entry_point: None,
            compilation_options: PipelineCompilationOptions {
                constants: &[("SIZE", SIZE as f64)],
                ..Default::default()
            },
            targets: &[Some(ColorTargetState {
                format: surface_config.format,
                blend: None,
                write_mask: Default::default(),
            })],
        }),
        cache: None,
        multiview_mask: None,
    });
    let mut vertex_bytes = Vec::new();
    vertices.append_bytes(&mut vertex_bytes);
    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &vertex_bytes,
        usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&indices),
        usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
    });

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: vertices.vertices.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    #[cfg(no_vertex_return)]
    const VERTEX_RETURN_FLAG: AccelerationStructureFlags = AccelerationStructureFlags::empty();
    #[cfg(not(no_vertex_return))]
    const VERTEX_RETURN_FLAG: AccelerationStructureFlags =
        AccelerationStructureFlags::ALLOW_RAY_HIT_VERTEX_RETURN;
    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE | VERTEX_RETURN_FLAG,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let blas = device.create_blas(
        &CreateBlasDescriptor {
            label: Some("test blas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE | VERTEX_RETURN_FLAG,
            update_mode: AccelerationStructureUpdateMode::Build,
        },
        BlasGeometrySizeDescriptors::Triangles {
            descriptors: vec![blas_size.clone()],
        },
    );

    *tlas.get_mut_single(0).unwrap() = Some(TlasInstance::new(
        &blas,
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        0,
        0xFF,
    ));

    let eye = Point3::new(0.5, 0.5, -1.0);
    let aim = Vector3::new(0.0, 0.0, 1.0);

    let proj = cgmath::perspective(cgmath::Deg(45.0f32), 1.0, 0.1, 200.0);
    let view = Matrix4::look_to_rh(eye, aim, Vector3::unit_y());
    //let proj = cgmath::perspective(cgmath::Deg(90.0f32), 1.0, 0.1, 200.0);
    //let view = Matrix4::look_at_rh(Point3::new(0.7, 0.9, 0.7), Point3::new(0.62, 0.66, 0.44), Vector3::unit_y());
    let cam = Camera::from_proj_view(proj.into(), view.into()).unwrap();

    let texture_unused = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    let mut textures = Vec::with_capacity(target_exe_num);
    for _ in 0..target_exe_num {
        textures.push(texture_unused.create_view(&TextureViewDescriptor::default()));
    }
    let mut texture_idx = 0;
    let mut texture_len = 0;

    let texture_back = device.create_texture(&TextureDescriptor {
        label: Some("background"),
        size: Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    encoder.build_acceleration_structures(
        iter::once(&BlasBuildEntry {
            blas: &blas,
            geometry: BlasGeometries::TriangleGeometries(vec![BlasTriangleGeometry {
                size: &blas_size,
                vertex_buffer: &vertex_buffer,
                first_vertex: 1,
                vertex_stride: mem::size_of::<[f32; 4]>() as BufferAddress,
                index_buffer: Some(&index_buffer),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }),
        iter::once(&tlas),
    );
    queue.submit(Some(encoder.finish()));

    let material_buf =
        device.create_buffer_init(&materials.buffer_descriptor(BufferUsages::COPY_DST));
    let material_indices_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&mat_indices),
        usage: BufferUsages::STORAGE,
    });

    #[cfg(no_vertex_return)]
    let material_bg = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(material_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::BufferArray(&[
                    material_indices_buf.as_entire_buffer_binding()
                ]),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::BufferArray(&[BufferBinding {
                    buffer: &vertex_buffer,
                    offset: 0,
                    size: None,
                }]),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::BufferArray(&[BufferBinding {
                    buffer: &index_buffer,
                    offset: 0,
                    size: None,
                }]),
            },
        ],
    });
    #[cfg(not(no_vertex_return))]
    let material_bg = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(material_buf.as_entire_buffer_binding()),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::BufferArray(&[
                    material_indices_buf.as_entire_buffer_binding()
                ]),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::AccelerationStructure(&tlas),
            },
        ],
    });
    let camera_buffer = device.create_buffer_init(&cam.buffer_descriptor(BufferUsages::empty()));

    let mut num_frames = 0u32;
    let mut old_percent = 0u8;
    let start = Instant::now();

    while !window.should_close() {
        if change_light_brightness {
            queue.write_buffer(
                &material_buf,
                (size_of::<Material>() * 3) as u64 + (size_of::<u32>() * 6) as u64,
                &(start.elapsed().as_secs_f32().cos().max(0.0) * POINT_BRIGHTNESS).to_ne_bytes(),
            );
        }
        let percent_done =
            (((texture_len as f32 / target_exe_num as f32) * 100.0).round() as u8).min(100);
        if old_percent < percent_done {
            eprintln!("{percent_done}%");
            old_percent = percent_done;
        }
        glfw.poll_events();
        let surface_texture = match surface.get_current_texture() {
            Ok(tex) => tex,
            Err(err) => match err {
                SurfaceError::Outdated | SurfaceError::Lost => {
                    surface_config.width = max(window.get_size().0 as u32, 1);
                    surface_config.height = max(window.get_size().1 as u32, 1);
                    surface.configure(&device, &surface_config);
                    continue;
                }
                err => panic!("{:?}", err),
            },
        };
        let texture = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let output_bg = buffers.create_bind_group(
            &device,
            &camera_buffer,
            &texture.create_view(&TextureViewDescriptor::default()),
            &texture_unused.create_view(&TextureViewDescriptor::default()),
            &texture_unused.create_view(&TextureViewDescriptor::default()),
            &texture_back.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..Default::default()
            }),
        );

        textures[texture_idx] = texture.create_view(&TextureViewDescriptor::default());
        texture_idx += 1;
        if texture_idx >= target_exe_num {
            texture_idx = 0;
        }
        if texture_len < target_exe_num {
            texture_len += 1;
        }
        let mut vec = Vec::with_capacity(target_exe_num);
        for view in &textures {
            vec.push(view)
        }
        let average_in = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &averaging_pipeline_in_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureViewArray(&vec),
            }],
        });
        {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

            {
                let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                comp_pass.set_pipeline(&compute_pipeline);
                comp_pass.set_bind_group(0, &material_bg, &[]);
                comp_pass.set_bind_group(1, &output_bg, &[]);
                comp_pass.set_bind_group(2, &texture_bg, &[]);
                // We input a different seed each frame.
                comp_pass.set_immediates(0, &num_frames.to_ne_bytes());
                let size = dispatch_size(SIZE, SIZE);
                comp_pass.dispatch_workgroups(size.width, size.height, 1);
                comp_pass.set_pipeline(&is_compute_pipeline);
                if importance_sampling {
                    comp_pass.dispatch_workgroups(size.width, size.height, 1);
                }
            }
            {
                let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                comp_pass.set_pipeline(&average_pipeline);
                comp_pass.set_bind_group(0, &average_in, &[]);
                comp_pass.set_bind_group(1, &oidn_state.staging_bg, &[]);
                comp_pass.set_immediates(0, &(texture_len as u32).to_ne_bytes());
                // using the same dispatch size as the other shaders
                let size = dispatch_size(SIZE, SIZE);
                comp_pass.dispatch_workgroups(size.width, size.height, 1);
            }
            // we use the non-temporal advance for spatial resampling because averaging reused data
            // is not a good idea
            buffers.advance_frame(
                &mut encoder,
                BufferType::MARKOV_CHAIN_SCREEN_SPACE
                    | BufferType::MARKOV_CHAIN_WORLD_SPACE
                    | BufferType::SPATIAL_RESAMPLING,
            );
            queue.submit(Some(encoder.finish()));
            #[cfg(feature = "denoise")]
            oidn_state.denoise(&device, &queue);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
            {
                let mut comp_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &surface_texture
                            .texture
                            .create_view(&TextureViewDescriptor::default()),
                        resolve_target: None,
                        ops: Operations::default(),
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                comp_pass.set_pipeline(&copy_pipeline);
                comp_pass.set_bind_group(0, &oidn_state.staging_bg, &[]);
                comp_pass.draw(0..3, 0..1);
            }
            queue.submit(Some(encoder.finish()));
        }
        surface_texture.present();
        if change_seed {
            num_frames += 1;
        }
    }
    drop(window);
    drop(surface);
}

struct OidnState<'a> {
    #[cfg(feature = "denoise")]
    filter: oidn::RayTracing<'a>,
    #[cfg(feature = "denoise")]
    buf: oidn::Buffer,
    #[cfg(feature = "denoise")]
    wgpu_map_read: wgpu::Buffer,
    #[cfg(feature = "denoise")]
    wgpu_map_write: wgpu::Buffer,
    #[cfg(feature = "denoise")]
    staging: wgpu::Buffer,
    staging_bgl: wgpu::BindGroupLayout,
    staging_bg: wgpu::BindGroup,
    phantom_data: PhantomData<&'a ()>,
}

impl<'a> OidnState<'a> {
    #[cfg(feature = "denoise")]
    fn new(wgpu_device: &wgpu::Device, device: &'a oidn::Device) -> Self {
        let buf_size_float = (SIZE * SIZE * 3) as usize;
        let wgpu_map_read = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readable"),
            size: (buf_size_float * mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let wgpu_map_write = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("writable"),
            size: (buf_size_float * mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::COPY_SRC | BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });
        let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (buf_size_float * mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bgl = wgpu_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg = wgpu_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: staging.as_entire_binding(),
            }],
        });
        let mut filter = oidn::RayTracing::new(&device);
        filter.image_dimensions(SIZE as usize, SIZE as usize);
        filter.filter_quality(oidn::Quality::Balanced);
        filter.hdr(true);
        filter.clean_aux(true);
        let buffer = device.create_buffer(&vec![0.0; buf_size_float]).unwrap();
        Self {
            filter,
            buf: buffer,
            wgpu_map_read,
            wgpu_map_write,
            staging,
            staging_bgl: bgl,
            staging_bg: bg,
            phantom_data: PhantomData,
        }
    }
    #[cfg(not(feature = "denoise"))]
    fn new(wgpu_device: &wgpu::Device) -> Self {
        let buf_size_float = (SIZE * SIZE * 3) as usize;
        let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (buf_size_float * mem::size_of::<f32>()) as BufferAddress,
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bgl = wgpu_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bg = wgpu_device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: staging.as_entire_binding(),
            }],
        });
        Self {
            staging_bgl: bgl,
            staging_bg: bg,
            phantom_data: PhantomData,
        }
    }
    #[cfg(feature = "denoise")]
    fn denoise(&mut self, wgpu_device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = wgpu_device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &self.staging,
            0,
            &self.wgpu_map_read,
            0,
            self.staging.size(),
        );
        queue.submit(Some(encoder.finish()));
        let (send, recv) = mpsc::channel();
        self.wgpu_map_read
            .slice(..)
            .map_async(MapMode::Read, move |res| {
                res.unwrap();
                send.send(()).unwrap()
            });
        wgpu_device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        recv.recv_timeout(Duration::from_secs(5)).unwrap();
        let (send, recv) = mpsc::channel();
        let mut data: Vec<f32> =
            bytemuck::cast_slice(&self.wgpu_map_read.slice(..).get_mapped_range()).to_vec();
        self.wgpu_map_read.unmap();
        self.buf.write(&data);
        self.filter.filter_in_place_buffer(&mut self.buf).unwrap();
        self.buf.read_to_slice(&mut data).unwrap();
        self.wgpu_map_write
            .slice(..)
            .map_async(MapMode::Write, move |res| {
                res.unwrap();
                send.send(()).unwrap()
            });
        wgpu_device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        recv.recv_timeout(Duration::from_secs(5)).unwrap();
        self.wgpu_map_write
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytemuck::cast_slice(&data));
        self.wgpu_map_write.unmap();
        let mut encoder = wgpu_device.create_command_encoder(&CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(
            &self.wgpu_map_write,
            0,
            &self.staging,
            0,
            self.staging.size(),
        );
        queue.submit(Some(encoder.finish()));
    }
}
