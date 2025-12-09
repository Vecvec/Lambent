use crate::Descriptor;
use cgmath::{Matrix4, SquareMatrix, Vector4};
use wgpu::util::BufferInitDescriptor;
use wgpu::BufferUsages;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub projection_inverse: [[f32; 4]; 4],
    pub view_inverse: [[f32; 4]; 4],
}

impl Camera {
    /// creates a camera from normal (not inverted) projection and view matrices
    pub fn from_proj_view(proj: [[f32; 4]; 4], view: [[f32; 4]; 4]) -> Result<Camera, CameraError> {
        let projection_inverse: [[f32; 4]; 4] =
            invert_matrix(proj).ok_or(CameraError::ProjectionUninvertable(proj))?;
        let view_inverse: [[f32; 4]; 4] =
            invert_matrix(view).ok_or(CameraError::ViewUninvertable(view))?;
        Ok(Camera {
            projection_inverse,
            view_inverse,
        })
    }
    /// updates a camera from normal (not inverted) projection and view matrices, either can be optional
    /// e.g. only changing fov
    pub fn update_from_proj_view(
        &mut self,
        proj: Option<[[f32; 4]; 4]>,
        view: Option<[[f32; 4]; 4]>,
    ) -> Result<(), CameraError> {
        if let Some(proj) = proj {
            self.projection_inverse =
                invert_matrix(proj).ok_or(CameraError::ProjectionUninvertable(proj))?;
        }
        if let Some(view) = view {
            self.view_inverse = invert_matrix(view).ok_or(CameraError::ViewUninvertable(view))?;
        }
        Ok(())
    }
}

impl Descriptor for Camera {
    fn buffer_descriptor(&self, further_usages: BufferUsages) -> BufferInitDescriptor<'_> {
        BufferInitDescriptor {
            label: Some("Camera"),
            contents: bytemuck::bytes_of(self),
            usage: BufferUsages::UNIFORM | further_usages,
        }
    }
}

#[derive(Debug)]
pub enum CameraError {
    ProjectionUninvertable([[f32; 4]; 4]),
    ViewUninvertable([[f32; 4]; 4]),
}

fn invert_matrix(mat: [[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    Some(
        Matrix4::from_cols(
            Vector4::from(mat[0]),
            Vector4::from(mat[1]),
            Vector4::from(mat[2]),
            Vector4::from(mat[3]),
        )
        .invert()?
        .into(),
    )
}
