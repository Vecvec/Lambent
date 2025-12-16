use std::borrow::Cow;

use crate::low_level::out_bgl;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, Buffer, BufferAddress, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipelineDescriptor, Device, Limits, PipelineLayoutDescriptor, PushConstantRange,
    ShaderModuleDescriptor, ShaderStages,
};

pub mod internal {
    use wesl::include_wesl;

    /// The source code for the end of frame processing shader (in wgsl).
    pub const FRAME_END_PROCESSING_SOURCE: &str = include_wesl!("end_of_frame");
}

bitflags::bitflags! {
    pub struct BufferType: u16 {
        // Need some of the world space stuff for the
        const MARKOV_CHAIN_SCREEN_SPACE = 1 << 0;
        #[cfg(feature = "wip-features")]
        const SPATIAL_RESAMPLING = 1 << 1;
        const MARKOV_CHAIN_WORLD_SPACE = 1 << 2;
    }
}

/// Matches markov chain in bindings.
#[repr(C)]
struct MarkovChain {
    _light_source: [f32; 3],
    _mean_cosine: f32,
    _weight_sum: f32,
    _num_samples: u32,
    _score: f32,
    _align: u32,
}

/// Matches atomic markov chain in bindings.
#[repr(C)]
struct AtomicMarkovChain {
    _light_source: [f32; 3],
    _mean_cosine: f32,
    _weight_sum: f32,
    _num_samples: u32,
    _score: f32,
    // align unneeded.
}

#[repr(C)]
struct WorldMarkovStorage {
    _secondary_hash: u32,
    _num_samples: u32,
    _radiance: [u32; 3],
    _lock: u32,
    _chain: AtomicMarkovChain,
    _timeout: u32,
}

struct TemporalBuffers {
    current: Buffer,
    old: Buffer,
}

impl TemporalBuffers {
    fn new(device: &Device, size: BufferAddress) -> Self {
        let current = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let old = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { current, old }
    }

    fn advance_frame(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.current, 0, &self.old, 0, self.current.size());
        encoder.clear_buffer(&self.current, 0, None);
    }

    fn advance_frame_no_temporal(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.current, 0, None);
        encoder.clear_buffer(&self.old, 0, None);
    }
}

// Right now there are lots of different possible importance samplers so the parts of this will likely change
// Therefore access is restricted to this so that changes to this are not breaking changes
/// Buffers for storing information between executions, if `wip-features` is enabled then this
/// contains importance sampling data. It is recommended to create this instead of making it yourself
/// as it may change in the future but will not be regarded as sem-var breaking.
pub struct DataBuffers {
    markov_chain_screen_space: TemporalBuffers,
    spatial_resampling: TemporalBuffers,
    markov_chain_world_space: TemporalBuffers,
    info: Buffer,
    #[cfg_attr(not(feature = "wip-features"), expect(dead_code))]
    gi_reservoir_processing_pipeline: wgpu::ComputePipeline,
    processing_bind_group: wgpu::BindGroup,
    limits: Limits,
    world_markov_chain_processing_pipeline: wgpu::ComputePipeline,
}

impl DataBuffers {
    pub fn new(
        device: &Device,
        mut width: u32,
        mut height: u32,
        world_space_buffer_size: u32,
        buffers: BufferType,
    ) -> Self {
        width = width.max(1);
        height = height.max(1);
        #[cfg(feature = "wip-features")]
        let spatial_buffers = buffers.contains(BufferType::SPATIAL_RESAMPLING);
        #[cfg(not(feature = "wip-features"))]
        let spatial_buffers = false;
        let size_spatial = if spatial_buffers {
            width as BufferAddress * height as BufferAddress
        } else {
            1
        } * size_of::<crate::importance_sampling::Reservoir>() as BufferAddress;
        let size_markov = if buffers.contains(BufferType::MARKOV_CHAIN_SCREEN_SPACE) {
            width as BufferAddress * height as BufferAddress
        } else {
            1
        } * size_of::<MarkovChain>() as BufferAddress;
        let size_markov_world = if buffers.contains(BufferType::MARKOV_CHAIN_WORLD_SPACE) {
            world_space_buffer_size.max(1) as BufferAddress
        } else {
            1
        } * size_of::<WorldMarkovStorage>() as BufferAddress;
        let info_size = width as BufferAddress
            * height as BufferAddress
            * size_of::<crate::importance_sampling::Info>() as BufferAddress;
        let info = device.create_buffer(&BufferDescriptor {
            label: None,
            size: info_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spatial_buffers = TemporalBuffers::new(device, size_spatial);
        let markov_buffers = TemporalBuffers::new(device, size_markov);
        let markov_world_buffers = TemporalBuffers::new(device, size_markov_world);

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("(phosph_rs internal) End of frame processing shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(internal::FRAME_END_PROCESSING_SOURCE)),
        });

        let processing_bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("(phosph_rs internal) End of frame processing bind group layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let processing_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("(phosph_rs internal) End of frame processing pipeline layout"),
            bind_group_layouts: &[&processing_bgl],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let gi_reservoir_processing_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("(phosph_rs internal) End of frame gi reservoir processor"),
                layout: Some(&processing_pipeline_layout),
                module: &shader,
                entry_point: Some("process_gi_reservoirs"),
                compilation_options: Default::default(),
                cache: None,
            });

        let world_markov_chain_processing_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("(phosph_rs internal) End of frame world space marcov chain processor"),
                layout: Some(&processing_pipeline_layout),
                module: &shader,
                entry_point: Some("process_world_markov_chains"),
                compilation_options: Default::default(),
                cache: None,
            });

        let processing_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("(phosph_rs internal) End of frame processing bind group"),
            layout: &processing_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: spatial_buffers.current.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: markov_world_buffers.current.as_entire_binding(),
                },
            ],
        });

        Self {
            markov_chain_screen_space: markov_buffers,
            spatial_resampling: spatial_buffers,
            markov_chain_world_space: markov_world_buffers,
            info,
            gi_reservoir_processing_pipeline,
            world_markov_chain_processing_pipeline,
            processing_bind_group,
            limits: device.limits(),
        }
    }

    /// Updates states to correct for a new frame, if a buffer is within
    /// `buffers_to_temporally_advance`, data will be moved to last frames buffer, otherwise it will
    /// be removed.
    pub fn advance_frame(
        &self,
        encoder: &mut CommandEncoder,
        buffers_to_temporally_advance: BufferType,
    ) {
        fn run_process_pipeline(
            buffers: &DataBuffers,
            encoder: &mut CommandEncoder,
            pipeline: &wgpu::ComputePipeline,
            mut num_workgroups: u32,
        ) {
            while num_workgroups != 0 {
                let execution_work_groups = buffers
                    .limits
                    .max_compute_workgroups_per_dimension
                    .min(num_workgroups.try_into().unwrap_or(<u32>::MAX));
                num_workgroups -= execution_work_groups;

                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &buffers.processing_bind_group, &[]);
                pass.set_push_constants(0, &num_workgroups.to_ne_bytes());
                pass.dispatch_workgroups(execution_work_groups, 1, 1);
            }
        }

        #[cfg(feature = "wip-features")]
        if buffers_to_temporally_advance.contains(BufferType::SPATIAL_RESAMPLING) {
            run_process_pipeline(
                self,
                encoder,
                &self.gi_reservoir_processing_pipeline,
                ((self.spatial_resampling.current.size()
                    / size_of::<crate::importance_sampling::Reservoir>() as u64)
                    as u32)
                    .div_ceil(256),
            );
            self.spatial_resampling.advance_frame(encoder);
        } else {
            self.spatial_resampling.advance_frame_no_temporal(encoder);
        }

        if buffers_to_temporally_advance.contains(BufferType::MARKOV_CHAIN_SCREEN_SPACE) {
            self.markov_chain_screen_space.advance_frame(encoder);
        } else {
            self.markov_chain_screen_space
                .advance_frame_no_temporal(encoder);
        }

        if buffers_to_temporally_advance.contains(BufferType::MARKOV_CHAIN_WORLD_SPACE) {
            run_process_pipeline(
                self,
                encoder,
                &self.world_markov_chain_processing_pipeline,
                ((self.markov_chain_world_space.current.size()
                    / size_of::<WorldMarkovStorage>() as u64) as u32)
                    .div_ceil(256),
            );
            encoder.copy_buffer_to_buffer(
                &self.markov_chain_world_space.current,
                0,
                &self.markov_chain_world_space.old,
                0,
                self.markov_chain_world_space.current.size(),
            );
        } else {
            self.markov_chain_world_space
                .advance_frame_no_temporal(encoder);
        }
    }

    pub fn create_bind_group(
        &self,
        device: &Device,
        cam_buf: &Buffer,
        tex_view: &wgpu::TextureView,
        tex_normal_view: &wgpu::TextureView,
        tex_albedo_view: &wgpu::TextureView,
        tex_background_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let out_bgl = out_bgl(device);
        device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &out_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(cam_buf.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(tex_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(tex_normal_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(tex_albedo_view),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(tex_background_view),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(
                        self.spatial_resampling.current.as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Buffer(
                        self.spatial_resampling.old.as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::Buffer(self.info.as_entire_buffer_binding()),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::Buffer(
                        self.markov_chain_screen_space
                            .current
                            .as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 9,
                    resource: BindingResource::Buffer(
                        self.markov_chain_screen_space
                            .old
                            .as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::Buffer(
                        self.markov_chain_world_space
                            .current
                            .as_entire_buffer_binding(),
                    ),
                },
                BindGroupEntry {
                    binding: 11,
                    resource: BindingResource::Buffer(
                        self.markov_chain_world_space.old.as_entire_buffer_binding(),
                    ),
                },
            ],
        })
    }
}
