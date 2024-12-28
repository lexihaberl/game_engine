use super::allocation::AllocatedBuffer;
use super::allocation::Allocator;
use super::device::Device;
use super::immediate_submit::ImmediateCommandData;
use ash::vk;
use nalgebra_glm as glm;
use std::path::Path;
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

#[derive(Debug, Copy, Clone)]
pub struct GeometricSurface {
    //idx of Surface in the buffer => we use one big buffer for whole mesh
    start_idx: usize,
    count: u32,
}

impl GeometricSurface {
    pub fn start_idx(&self) -> usize {
        self.start_idx
    }
    pub fn count(&self) -> u32 {
        self.count
    }
}

pub struct MeshAsset {
    #[allow(dead_code)]
    name: String,
    surfaces: Vec<GeometricSurface>,
    buffers: GPUMeshBuffers,
}

impl MeshAsset {
    pub fn load_gltf(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        immediate_command_data: &ImmediateCommandData,
        file_path: &Path,
        overwrite_color_with_normals: bool,
    ) -> Result<Vec<Self>, gltf::Error> {
        log::info!("Loading GLTF from file: {:?}", file_path);

        let (gltf, buffers, _) = gltf::import(file_path)?;

        let mut meshes = Vec::new();
        let mut indices = Vec::new();
        let mut vertices = Vec::new();
        for mesh in gltf.meshes() {
            // we store per mesh indices/vertices => clear them for each mesh
            indices.clear();
            vertices.clear();
            let mut surfaces = Vec::new();

            let mesh_name = mesh.name().unwrap_or("Unnamed Mesh");
            log::debug!("Loading mesh: {}", mesh_name);

            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let start_idx = indices.len();
                let initial_vtx = vertices.len();
                let mut count = 0;

                if let Some(iter) = reader.read_indices() {
                    let iter = iter.into_u32();
                    indices.reserve(iter.len() + indices.len());
                    count = iter.len() as u32;
                    for index in iter {
                        indices.push(index + initial_vtx as u32);
                    }
                }
                surfaces.push(GeometricSurface { start_idx, count });

                match reader.read_positions() {
                    Some(iter) => {
                        vertices.reserve(iter.len() + vertices.len());
                        for vertex_position in iter {
                            vertices.push(Vertex::new(
                                glm::vec3(
                                    vertex_position[0],
                                    vertex_position[1],
                                    vertex_position[2],
                                ),
                                0.0,
                                glm::vec3(0.0, 0.0, 0.0),
                                0.0,
                                glm::vec4(1.0, 1.0, 1.0, 1.0),
                            ));
                        }
                    }
                    None => panic!("No positions found in mesh"),
                }

                match reader.read_normals() {
                    Some(iter) => {
                        for (i, vertex_normal) in iter.enumerate() {
                            vertices[i + initial_vtx].normal =
                                glm::vec3(vertex_normal[0], vertex_normal[1], vertex_normal[2]);
                        }
                    }
                    None => log::warn!("No normals found in mesh"),
                }

                match reader.read_tex_coords(0) {
                    Some(iter) => {
                        let iter = iter.into_f32();
                        for (i, vertex_uv) in iter.enumerate() {
                            vertices[i + initial_vtx].uv_x = vertex_uv[0];
                            vertices[i + initial_vtx].uv_y = vertex_uv[1];
                        }
                    }
                    None => log::warn!("No UVs found in mesh"),
                }

                match reader.read_colors(0) {
                    Some(iter) => {
                        let iter = iter.into_rgba_f32();
                        for (i, vertex_color) in iter.enumerate() {
                            vertices[i + initial_vtx].color = glm::vec4(
                                vertex_color[0],
                                vertex_color[1],
                                vertex_color[2],
                                vertex_color[3],
                            );
                        }
                    }
                    None => log::warn!(
                        "No colors found in mesh {} loaded from file {:?}",
                        mesh_name,
                        file_path
                    ),
                }
            }
            if overwrite_color_with_normals {
                for vertex in &mut vertices {
                    vertex.color =
                        glm::vec4(vertex.normal.x, vertex.normal.y, vertex.normal.z, 1.0);
                }
            }
            let new_mesh = MeshAsset {
                name: mesh_name.to_string(),
                surfaces,
                buffers: GPUMeshBuffers::upload_mesh(
                    device.clone(),
                    allocator.clone(),
                    &indices,
                    &vertices,
                    immediate_command_data,
                ),
            };
            meshes.push(new_mesh);
        }
        Ok(meshes)
    }

    pub fn buffers(&self) -> &GPUMeshBuffers {
        &self.buffers
    }

    pub fn surfaces(&self) -> &Vec<GeometricSurface> {
        &self.surfaces
    }

    #[allow(dead_code)]
    pub fn name(&self) -> &str {
        &self.name
    }
}

pub struct Sampler {
    device: Arc<Device>,
    sampler: vk::Sampler,
}

impl Sampler {
    pub fn new(device: Arc<Device>, min_filter: vk::Filter, mag_filter: vk::Filter) -> Self {
        let create_info = vk::SamplerCreateInfo {
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            mag_filter,
            min_filter,
            ..Default::default()
        };
        let sampler = device.create_sampler(&create_info);
        Self { device, sampler }
    }

    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        log::debug!("Dropping Sampler");
        self.device.destroy_sampler(self.sampler);
    }
}
