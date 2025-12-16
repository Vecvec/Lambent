use crate::low_level::texture_bgl;
use std::fs::read;
use std::mem;
use std::num::NonZeroU32;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindingResource, Device,
    Extent3d, Origin3d, Queue, Sampler, SamplerDescriptor, TexelCopyBufferLayout,
    TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

#[derive(Default)]
pub struct TextureLoader {
    images: Vec<image::RgbaImage>,
}

impl TextureLoader {
    pub fn new() -> Self {
        Self { images: Vec::new() }
    }

    /// Loads an image from the raw bytes, useful for images whose file name are not known at compile-time
    ///
    /// For compile-time known images use [TextureLoader::load_from_bytes]
    pub fn load(&mut self, tex: &str) -> Result<(), TextureError> {
        let diffuse = read(tex).map_err(TextureError::FileError)?;
        self.load_from_bytes(&diffuse)
    }

    /// Loads an image from the raw bytes, useful for [include_bytes] or dynamic program changeable textures
    pub fn load_from_bytes(&mut self, tex: &[u8]) -> Result<(), TextureError> {
        self.images.push(
            image::load_from_memory(tex)
                .map_err(TextureError::Image)?
                .to_rgba8(),
        );
        Ok(())
    }

    /// Converts the texture loader to the wgpu textures
    pub fn create_textures(
        &self,
        device: &Device,
        queue: &Queue,
        additional_texture_usages: TextureUsages,
    ) -> WgpuTextures {
        let mut textures = Vec::with_capacity(self.images.len());
        let mut views = Vec::with_capacity(self.images.len());
        for image in &self.images {
            let texture = device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d {
                    width: image.width(),
                    height: image.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | additional_texture_usages,
                view_formats: &[],
            });
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                    aspect: TextureAspect::All,
                },
                image.as_raw(),
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(image.width() * mem::size_of::<[u8; 4]>() as u32),
                    rows_per_image: None,
                },
                Extent3d {
                    width: image.width(),
                    height: image.height(),
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&TextureViewDescriptor::default());
            textures.push(texture);
            views.push(view);
        }
        WgpuTextures { textures, views }
    }
}

#[derive(Debug)]
pub enum TextureError {
    Image(image::error::ImageError),
    FileError(std::io::Error),
}

#[non_exhaustive]
#[derive(Clone)]
pub struct WgpuTextures {
    pub textures: Vec<Texture>,
    pub views: Vec<TextureView>,
}

impl WgpuTextures {
    pub fn from_textures(textures: Vec<Texture>) -> Self {
        let views = textures
            .iter()
            .map(|tex| tex.create_view(&TextureViewDescriptor::default()))
            .collect::<Vec<_>>();
        Self { textures, views }
    }

    pub fn add_texture(&mut self, tex: Texture) {
        let view = tex.create_view(&TextureViewDescriptor::default());
        self.textures.push(tex);
        self.views.push(view);
    }
}

fn create_blank_recolour(device: &Device, queue: &Queue) -> TextureView {
    let tex = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d::default(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: Default::default(),
            aspect: TextureAspect::All,
        },
        &[1, 1, 1, 1],
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(mem::size_of::<[u8; 4]>() as u32),
            rows_per_image: None,
        },
        Extent3d::default(),
    );
    tex.create_view(&TextureViewDescriptor::default())
}

pub fn bind_group_from_textures(
    device: &Device,
    queue: &Queue,
    diffuse_tex: &WgpuTextures,
    emission_tex: &WgpuTextures,
    attribute_tex: &WgpuTextures,
    recolour_tex: Option<TextureView>,
    sampler: Option<&Sampler>,
) -> (BindGroup, BindGroupLayout) {
    let bgl = texture_bgl(
        device,
        [
            NonZeroU32::new(diffuse_tex.textures.len() as u32).unwrap(),
            NonZeroU32::new(emission_tex.textures.len() as u32).unwrap(),
            NonZeroU32::new(attribute_tex.textures.len() as u32).unwrap(),
        ],
    );

    let bg = bind_group_from_textures_bindgroup_layout(
        device,
        queue,
        diffuse_tex,
        emission_tex,
        attribute_tex,
        recolour_tex,
        sampler,
        &bgl,
    );
    (bg, bgl)
}

pub fn bind_group_from_textures_bindgroup_layout(
    device: &Device,
    queue: &Queue,
    diffuse_tex: &WgpuTextures,
    emission_tex: &WgpuTextures,
    attribute_tex: &WgpuTextures,
    recolour_tex: Option<TextureView>,
    sampler: Option<&Sampler>,
    layout: &BindGroupLayout,
) -> BindGroup {
    let sampler_storage;
    let sampler = match sampler {
        None => {
            let sampler = device.create_sampler(&SamplerDescriptor::default());
            sampler_storage = sampler;
            &sampler_storage
        }
        Some(sampler) => sampler,
    };

    let recolour_tex = recolour_tex.unwrap_or_else(|| create_blank_recolour(device, queue));
    let mut diffuse_views = Vec::with_capacity(diffuse_tex.views.len());
    for view in diffuse_tex.views.iter() {
        diffuse_views.push(view);
    }
    let mut emission_views = Vec::with_capacity(diffuse_tex.views.len());
    for view in emission_tex.views.iter() {
        emission_views.push(view);
    }
    let mut attribute_views = Vec::with_capacity(diffuse_tex.views.len());
    for view in attribute_tex.views.iter() {
        attribute_views.push(view);
    }
    let bg = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureViewArray(&diffuse_views),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::TextureViewArray(&emission_views),
            },
            BindGroupEntry {
                binding: 3,
                resource: BindingResource::TextureViewArray(&attribute_views),
            },
            BindGroupEntry {
                binding: 4,
                resource: BindingResource::TextureView(&recolour_tex),
            },
        ],
    });
    bg
}
