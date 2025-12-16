use crate::camera::Camera;
#[cfg(feature = "wip-features")]
use crate::importance_sampling::SpatialResampling;
use crate::low_level::RayTracingShaderDST;
use crate::textures::TextureLoader;
use crate::{Descriptor, Material, MaterialType, Vertices, debug, dispatch_size, path_tracing, textures};
use crate::{BufferType, DataBuffers};
use cgmath::{ElementWise, Matrix4, Point3, Vector3};
use futures::executor::block_on;
use glfw::{ClientApiHint, Glfw, PWindow, WindowHint, WindowMode};
use image::EncodableLayout;
use log::LevelFilter;
use std::cmp::max;
use std::num::NonZeroU32;
use std::time::Instant;
use std::{iter, mem};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    AccelerationStructureFlags, AccelerationStructureGeometryFlags,
    AccelerationStructureUpdateMode, Adapter, Backends, BindGroupDescriptor, BindGroupEntry,
    BindingResource, BlasBuildEntry, BlasGeometries, BlasGeometrySizeDescriptors,
    BlasTriangleGeometry, BlasTriangleGeometrySizeDescriptor, BufferAddress, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, CreateBlasDescriptor, CreateTlasDescriptor,
    DeviceDescriptor, ExperimentalFeatures, Extent3d, Features, IndexFormat, Instance,
    InstanceDescriptor, Origin3d, PresentMode, Queue, RequestDeviceError, Surface, SurfaceError,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
    TlasInstance, VertexFormat,
};

const SIZE: (u32, u32) = (1280, 720);

#[test]
fn test() {
    env_logger::init();

    log::set_max_level(LevelFilter::Trace);
    let shaders: &'static [&'static dyn RayTracingShaderDST] = &[
        // proper rt shaders
        &path_tracing::High,
        &path_tracing::Medium,
        &path_tracing::Low,
        // debug shaders
        &debug::FrontFace,
        &debug::Reflectance,
        &debug::Tangent,
    ];

    let positions = [[1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
    let materials = [
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0, 0.0],
            0,
            Some(0),
            Some(0),
            1.0,
            Some(crate::refractive_indices::DIAMOND),
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[3],
            positions[2],
            positions[1],
            [0.0, 0.0],
            0,
            Some(0),
            Some(0),
            1.0,
            Some(crate::refractive_indices::DIAMOND),
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[0],
            positions[2],
            positions[1],
            [0.0, 0.0],
            1,
            Some(1),
            None,
            1.0,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[1],
            positions[3],
            positions[2],
            [0.0, 0.0],
            1,
            Some(1),
            None,
            1.0,
            None,
            MaterialType::Diffuse,
        ),
        Material::new(
            positions[2],
            positions[3],
            positions[0],
            [0.0, 0.0],
            0,
            Some(0),
            Some(0),
            1.0,
            None,
            MaterialType::Metallic,
        ),
        Material::new(
            positions[1],
            positions[0],
            positions[3],
            [0.0, 0.0],
            0,
            Some(0),
            Some(0),
            1.0,
            None,
            MaterialType::Metallic,
        ),
        Material::new(
            positions[0],
            positions[1],
            positions[2],
            [0.0, 0.0],
            1,
            None,
            None,
            1.0,
            Some(crate::refractive_indices::WATER),
            MaterialType::Transparent,
        ),
        Material::new(
            positions[1],
            positions[3],
            positions[2],
            [0.0, 0.0],
            1,
            None,
            None,
            1.0,
            Some(crate::refractive_indices::WATER),
            MaterialType::Transparent,
        ),
    ];
    let material_indices = [0, 1, 2, 3, 4, 5, 6, 7];
    let vertices = Vertices { geometry_stride: 0, vertices: vec![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
    ]};
    let indices = [
        0, 1, 2, 3, 2, 1, 0, 4, 1, 1, 5, 4, 0, 2, 4, 6, 4, 2, 2, 3, 6, 7, 6, 3,
    ];
    let mut glfw = glfw::init_no_callbacks().unwrap();
    // on some platforms this fixes crashes
    glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
    let (mut window, _) = glfw
        .create_window(SIZE.0, SIZE.1, "raytracing tests", WindowMode::Windowed)
        .unwrap();
    for &shader in shaders {
        exe_shader(
            shader,
            &vertices,
            &indices,
            &materials,
            &material_indices,
            &mut glfw,
            &mut window,
            false,
        );
        #[cfg(feature = "wip-features")]
        exe_shader(
            shader,
            &vertices,
            &indices,
            &materials,
            &material_indices,
            &mut glfw,
            &mut window,
            true,
        );
    }
}

fn exe_shader(
    shader: &dyn RayTracingShaderDST,
    vertices: &Vertices,
    indices: &[u32],
    materials: &[Material],
    material_indices: &[u32],
    glfw: &mut Glfw,
    window: &mut PWindow,
    run_is: bool,
) {
    println!();
    let mut skipped = Vec::new();
    let instance = Instance::new(&InstanceDescriptor::default());
    let surface = instance.create_surface(window.render_context()).unwrap();
    let adapters = instance.enumerate_adapters(Backends::default());
    for adapter in &adapters {
        match run_shader(
            shader,
            adapter,
            &surface,
            vertices,
            indices,
            materials,
            material_indices,
            glfw,
            window,
            run_is,
        ) {
            Ok(_) => {}
            Err(ExcErr::Other(string)) => {
                log::error!(
                    "Error on {}:\n    {string}, {:?}",
                    adapter.get_info().name,
                    adapter.get_info().backend
                );
            }
            Err(ExcErr::Device(err)) => skipped.push((
                format!(
                    "{} ({:?})",
                    adapter.get_info().name,
                    adapter.get_info().backend
                ),
                err,
            )),
        }
    }
    let mut string = String::new();
    fmt_skipped(skipped, &mut string);
    log::debug!(
        "Executed shader on all supported devices, skipped: \n {}",
        string
    );
}

fn fmt_skipped(skipped: Vec<(String, RequestDeviceError)>, string: &mut String) {
    for device in skipped {
        string.push_str(&format!("{} due to {} \n", device.0, device.1));
    }
}

#[derive(Debug)]
enum ExcErr {
    Device(RequestDeviceError),
    Other(String),
}

#[allow(clippy::too_many_arguments)]
fn run_shader(
    shader: &dyn RayTracingShaderDST,
    adapter: &Adapter,
    surface: &Surface,
    vertices: &Vertices,
    indices: &[u32],
    materials: &[Material],
    material_indices: &[u32],
    glfw: &mut Glfw,
    window: &mut PWindow,
    run_is: bool,
) -> Result<(), ExcErr> {
    let (device, queue) = block_on(
        adapter.request_device(&DeviceDescriptor {
            label: Some(adapter.get_info().name.as_str()),
            required_features: shader.features() | Features::BGRA8UNORM_STORAGE,
            required_limits: shader
                .limits()
                .using_minimum_supported_acceleration_structure_values(),
            memory_hints: Default::default(),
            trace: Default::default(),
            experimental_features: unsafe { ExperimentalFeatures::enabled() },
        }),
    )
    .map_err(ExcErr::Device)?;
    log::info!(
        "Found device:\n   {}, {:?}",
        adapter.get_info().name,
        adapter.get_info().backend
    );

    let shader = shader.dyn_ray_tracer(&device);
    let mut surface_config = surface
        .get_default_config(adapter, SIZE.0, SIZE.1)
        .ok_or(ExcErr::Other("failed to get default config".to_string()))?;
    surface_config.format = TextureFormat::Bgra8Unorm;
    surface_config.usage |= TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING;
    surface_config.present_mode = PresentMode::AutoNoVsync;
    surface.configure(&device, &surface_config);

    let mut loader_diffuse = TextureLoader::new();
    loader_diffuse
        .load_from_bytes(include_bytes!("diffuse.png"))
        .expect("these are valid PNGs so this should not fail");
    loader_diffuse
        .load_from_bytes(include_bytes!("pure_white.png"))
        .expect("these are valid PNGs so this should not fail");
    let diffuse_textures = loader_diffuse.create_textures(&device, &queue, TextureUsages::empty());
    let mut loader_emission = TextureLoader::new();
    loader_emission
        .load_from_bytes(include_bytes!("emission_partial.png"))
        .expect("these are valid PNGs so this should not fail");
    loader_emission
        .load_from_bytes(include_bytes!("emission.png"))
        .expect("these are valid PNGs so this should not fail");
    let emission_textures =
        loader_emission.create_textures(&device, &queue, TextureUsages::empty());
    let mut loader_attribute = TextureLoader::new();
    loader_attribute
        .load_from_bytes(include_bytes!("attributes.png"))
        .expect("these are valid PNGs so this should not fail");
    let attribute_textures =
        loader_attribute.create_textures(&device, &queue, TextureUsages::empty());

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

    load_background(&queue, &texture_back);

    let mut buffers = DataBuffers::new(
        &device,
        window.get_size().0 as u32,
        window.get_size().1 as u32,
        1_000,
        BufferType::all(),
    );

    let (texture_bg, _) = textures::bind_group_from_textures(
        &device,
        &queue,
        &diffuse_textures,
        &emission_textures,
        &attribute_textures,
        None,
        None,
    );
    let samples: u32 = if run_is { 1 } else { 8 };

    #[cfg(feature = "wip-features")]
    let layout = crate::low_level::pipeline_layout(
        &device,
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(1).unwrap(),
        &[],
    );

    let compute_pipeline = shader.create_pipeline(
        NonZeroU32::new(1).unwrap(),
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(2).unwrap(),
        NonZeroU32::new(1).unwrap(),
        &[
            ("SAMPLES", samples as f64),
            // this is a test, these are just here to check all shaders contain them
            ("T_MIN", 0.01),
            ("T_MAX", 10.0),
        ],
    );

    #[cfg(feature = "wip-features")]
    let is_shader = SpatialResampling::create_shader();
    #[cfg(feature = "wip-features")]
    let is_shader = device.create_shader_module(is_shader.descriptor());
    #[cfg(feature = "wip-features")]
    let is_compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &is_shader,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &[],
            ..Default::default()
        },
        cache: None,
    });

    let mut bytes = Vec::new();
    vertices.append_bytes(&mut bytes);

    let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: &bytes,
        usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
    });
    let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(indices),
        usage: BufferUsages::BLAS_INPUT | BufferUsages::STORAGE,
    });

    let blas_size = BlasTriangleGeometrySizeDescriptor {
        vertex_format: VertexFormat::Float32x3,
        vertex_count: vertices.vertices.len() as u32,
        index_format: Some(IndexFormat::Uint32),
        index_count: Some(indices.len() as u32),
        flags: AccelerationStructureGeometryFlags::OPAQUE,
    };
    let mut tlas = device.create_tlas(&CreateTlasDescriptor {
        label: None,
        max_instances: 1,
        flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: AccelerationStructureUpdateMode::Build,
    });
    let blas = device.create_blas(
        &CreateBlasDescriptor {
            label: Some("test blas"),
            flags: AccelerationStructureFlags::PREFER_FAST_TRACE,
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

    const HEIGHT: f32 = 0.8;
    let start_eye = Point3::new(0.5, HEIGHT, 0.5);
    let start_centre = Point3::new(0.0, 0.0, 0.0);
    let end_eye = Point3::new(1.0, HEIGHT, 1.0);
    let end_centre = Point3::new(1.0, 1.1, 1.0);

    // near and far are arbitrary, just something that might be used for rendering
    let proj = cgmath::perspective(
        cgmath::Deg(60.0f32),
        SIZE.0 as f32 / SIZE.1 as f32,
        0.1,
        200.0,
    );
    let view = Matrix4::look_at_rh(start_eye, start_centre, Vector3::unit_y());
    let mut cam = Camera::from_proj_view(proj.into(), view.into()).unwrap();

    let texture_unused = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: SIZE.0,
            height: SIZE.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
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
        device.create_buffer_init(&materials.buffer_descriptor(BufferUsages::empty()));
    let material_indices_buf = device.create_buffer_init(&BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(material_indices),
        usage: BufferUsages::STORAGE,
    });

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
                binding: 4,
                resource: BindingResource::BufferArray(&[
                    vertex_buffer.as_entire_buffer_binding()
                ]),
            },
            BindGroupEntry {
                binding: 5,
                resource: BindingResource::BufferArray(&[
                    index_buffer.as_entire_buffer_binding()
                ]),
            },
        ],
    });

    let start = Instant::now();
    let timer = Instant::now();
    let mut num_frames = 0u32;
    let mut elapsed = 0;
    const MAX_ELAPSED: u64 = 10;
    while elapsed < MAX_ELAPSED {
        if window.should_close() {
            log::info!("Skipping test!");
            window.set_should_close(false);
            return Ok(());
        }
        let time_blend = (start.elapsed().as_secs_f32() / (MAX_ELAPSED as f32)).min(1.0);
        let eye = (end_eye * time_blend).add_element_wise(start_eye * (1.0 - time_blend));
        let centre = (end_centre * time_blend).add_element_wise(start_centre * (1.0 - time_blend));
        let view = Matrix4::look_at_rh(eye, centre, Vector3::unit_y());
        let _ = cam.update_from_proj_view(None, Some(view.into()));

        let camera_buffer =
            device.create_buffer_init(&cam.buffer_descriptor(BufferUsages::empty()));
        glfw.poll_events();
        let surface_texture = match surface.get_current_texture() {
            Ok(tex) => tex,
            Err(err) => match err {
                SurfaceError::Outdated | SurfaceError::Lost => {
                    surface_config.width = max(window.get_size().0 as u32, 1);
                    surface_config.height = max(window.get_size().1 as u32, 1);
                    surface.configure(&device, &surface_config);

                    buffers = DataBuffers::new(
                        &device,
                        window.get_size().0 as u32,
                        window.get_size().1 as u32,
                        1_000,
                        BufferType::all(),
                    );
                    continue;
                }
                _ => return Err(ExcErr::Other(format!("Failed to get surface: {err}"))),
            },
        };
        let output_tex = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: surface_config.width,
                height: surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output_bg = buffers.create_bind_group(
            &device,
            &camera_buffer,
            &output_tex.create_view(&TextureViewDescriptor::default()),
            &texture_unused.create_view(&TextureViewDescriptor::default()),
            &texture_unused.create_view(&TextureViewDescriptor::default()),
            &texture_back.create_view(&TextureViewDescriptor {
                dimension: Some(TextureViewDimension::Cube),
                ..Default::default()
            }),
        );

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

        {
            let mut comp_pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            comp_pass.set_pipeline(&compute_pipeline);
            comp_pass.set_bind_group(0, &material_bg, &[]);
            comp_pass.set_bind_group(1, &output_bg, &[]);
            comp_pass.set_bind_group(2, &texture_bg, &[]);
            // not technically necessary, but this demos making seed random, also eye (kinda) and monitor denoises bc of the way it works
            comp_pass.set_push_constants(0, &num_frames.to_ne_bytes());
            let size = dispatch_size(surface_config.width, surface_config.height);
            comp_pass.dispatch_workgroups(size.width, size.height, 1);
            #[cfg(feature = "wip-features")]
            if run_is {
                comp_pass.set_pipeline(&is_compute_pipeline);
                comp_pass.dispatch_workgroups(size.width, size.height, 1);
            }
        }

        buffers.advance_frame(&mut encoder, BufferType::all());

        let blit = wgpu::util::TextureBlitter::new(&device, surface_config.format);

        blit.copy(
            &device,
            &mut encoder,
            &output_tex.create_view(&TextureViewDescriptor::default()),
            &surface_texture
                .texture
                .create_view(&TextureViewDescriptor::default()),
        );

        queue.submit(Some(encoder.finish()));
        surface_texture.present();
        let new_time = start.elapsed().as_secs();
        if new_time > elapsed {
            elapsed = new_time;
        }
        num_frames += 1;
    }
    log::info!(
        "Average fps: {}",
        1.0 / (timer.elapsed().as_secs_f64() / num_frames as f64)
    );
    Ok(())
}

fn load_background(queue: &Queue, tex: &Texture) {
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 0 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("plus_x.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 1 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("minus_x.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 2 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("plus_y.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 3 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("minus_y.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 4 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("plus_z.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: Origin3d { x: 0, y: 0, z: 5 },
            aspect: TextureAspect::All,
        },
        image::load_from_memory(include_bytes!("minus_z.png"))
            .unwrap()
            .to_rgba8()
            .as_bytes(),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some((64 * mem::size_of::<[u8; 4]>()) as u32),
            rows_per_image: Some(64),
        },
        Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
}
