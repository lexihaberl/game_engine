use super::allocation::AllocatedBuffer;
use super::allocation::Allocator;
use super::device::Device;
use super::immediate_submit::ImmediateCommandData;
use ash::vk;
use nalgebra_glm as glm;
use std::sync::Arc;
use std::sync::Mutex;

#[repr(C)]
#[derive(Debug, bytemuck::NoUninit, Copy, Clone)]
pub struct Vertex {
    position: glm::Vec3,
    uv_x: f32,
    normal: glm::Vec3,
    uv_y: f32,
    color: glm::Vec4,
}

impl Vertex {
    pub fn new(
        position: glm::Vec3,
        uv_x: f32,
        normal: glm::Vec3,
        uv_y: f32,
        color: glm::Vec4,
    ) -> Self {
        Self {
            position,
            uv_x,
            normal,
            uv_y,
            color,
        }
    }
}

#[repr(C)]
pub struct GPUMeshBuffers {
    index_buffer: AllocatedBuffer,
    vertex_buffer: AllocatedBuffer,
    vertex_buffer_address: vk::DeviceAddress,
}

impl GPUMeshBuffers {
    pub fn upload_mesh(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        indices: &[u32],
        vertices: &[Vertex],
        immediate_command: &ImmediateCommandData,
    ) -> Self {
        let vertex_buffer_size = std::mem::size_of_val(vertices);
        let vertex_buffer = AllocatedBuffer::new(
            device.clone(),
            allocator.clone(),
            "Vertex Buffer",
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vertex_buffer_size as vk::DeviceSize,
            gpu_allocator::MemoryLocation::GpuOnly,
        );
        let buffer_device_address = vertex_buffer.get_device_address();

        let index_buffer_size = std::mem::size_of_val(indices);
        let index_buffer = AllocatedBuffer::new(
            device.clone(),
            allocator.clone(),
            "Index Buffer",
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            index_buffer_size as vk::DeviceSize,
            gpu_allocator::MemoryLocation::GpuOnly,
        );

        let mut staging_buffer = AllocatedBuffer::new(
            device,
            allocator,
            "Staging Buffer",
            vk::BufferUsageFlags::TRANSFER_SRC,
            (vertex_buffer_size + index_buffer_size) as vk::DeviceSize,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );

        staging_buffer.copy_from_slice(vertices, 0);
        staging_buffer.copy_from_slice(indices, vertex_buffer_size);

        immediate_command.immediate_submit(|device, command_buffer| {
            let vertex_copy = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: vertex_buffer_size as vk::DeviceSize,
            };
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.buffer(),
                vertex_buffer.buffer(),
                &[vertex_copy],
            );
            let index_copy = vk::BufferCopy {
                src_offset: vertex_buffer_size as vk::DeviceSize,
                dst_offset: 0,
                size: index_buffer_size as vk::DeviceSize,
            };
            device.cmd_copy_buffer(
                command_buffer,
                staging_buffer.buffer(),
                index_buffer.buffer(),
                &[index_copy],
            );
        });

        Self {
            index_buffer,
            vertex_buffer,
            vertex_buffer_address: buffer_device_address,
        }
    }

    pub fn vertex_buffer_address(&self) -> vk::DeviceAddress {
        self.vertex_buffer_address
    }

    pub fn index_buffer(&self) -> vk::Buffer {
        self.index_buffer.buffer()
    }
}

#[repr(C)]
#[derive(Debug, bytemuck::NoUninit, Copy, Clone)]
pub struct GPUDrawPushConstants {
    pub world_matrix: glm::Mat4,
    pub device_address: vk::DeviceAddress,
}

impl GPUDrawPushConstants {
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
