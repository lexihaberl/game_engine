use super::allocation::AllocatedBuffer;
use super::allocation::AllocatedImage;
use super::allocation::Allocator;
use super::descriptor::DescriptorAllocator;
use super::descriptor::DescriptorLayoutBuilder;
use super::descriptor::DescriptorSetLayout;
use super::descriptor::DescriptorWriter;
use super::device::Device;
use super::immediate_submit::ImmediateCommandData;
use super::pipelines::GraphicsPipeline;
use super::pipelines::GraphicsPipelineBuilder;
use super::shader::ShaderModule;
use ash::vk;
use nalgebra_glm as glm;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;

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

#[derive(Clone)]
pub struct GeometricSurface {
    //idx of Surface in the buffer => we use one big buffer for whole mesh
    start_idx: usize,
    count: u32,
    //TODO: remove Option once we implement material loading from GLTF
    material: Arc<MaterialInstance>,
}

impl GeometricSurface {
    pub fn start_idx(&self) -> usize {
        self.start_idx
    }
    pub fn count(&self) -> u32 {
        self.count
    }
    pub fn material(&self) -> Arc<MaterialInstance> {
        self.material.clone()
    }

    #[allow(dead_code)]
    pub fn set_material(&mut self, material: Arc<MaterialInstance>) {
        self.material = material;
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
        default_material: Option<Arc<MaterialInstance>>,
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
                if let Some(default_material) = default_material.as_ref() {
                    let material = default_material.clone();
                    let surface = GeometricSurface {
                        start_idx,
                        count,
                        material,
                    };
                    surfaces.push(surface);
                } else {
                    todo!(
                        "No default material provided and material loading is not yet implemented"
                    );
                }

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

#[derive(Debug, Clone, Copy)]
pub enum MaterialPass {
    MainColor,
    Transparent,
    Other,
}

pub struct MaterialPipeline {
    pipeline: GraphicsPipeline,
    layout: vk::PipelineLayout,
}

pub struct MaterialInstance {
    pipeline: Arc<MaterialPipeline>,
    material_set: vk::DescriptorSet,
    pass_type: MaterialPass,
}
impl MaterialInstance {
    pub fn bind_pipeline(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.pipeline.bind(command_buffer);
    }
}

impl Clone for MaterialInstance {
    fn clone(&self) -> Self {
        MaterialInstance {
            pipeline: self.pipeline.clone(),
            material_set: self.material_set,
            pass_type: self.pass_type,
        }
    }
}

pub struct MaterialResources {
    pub color_image: Arc<AllocatedImage>,
    pub color_sampler: Arc<Sampler>,
    pub metal_rough_image: Arc<AllocatedImage>,
    pub metal_rough_sampler: Arc<Sampler>,
    pub data_buffer: AllocatedBuffer,
    pub data_buffer_offset: u64,
}

#[derive(Copy, Clone)]
#[repr(C)]
// will be used for uniform buffers later on -> minimum alignment of 256 supported by most GPUS
pub struct MaterialConstants {
    pub color_factors: glm::Vec4,
    pub metal_rough_factors: glm::Vec4,
    // align to 256 bytes
    pub extra_data: [glm::Vec4; 14],
}

pub struct GLTFMetallicRoughness<'a> {
    opaque_pipeline: Arc<MaterialPipeline>,
    transparent_pipeline: Arc<MaterialPipeline>,
    material_layout: DescriptorSetLayout,
    writer: DescriptorWriter<'a>,
}

impl<'a> GLTFMetallicRoughness<'a> {
    pub fn build_pipeline(
        device: Arc<Device>,
        draw_image_format: vk::Format,
        depth_image_format: vk::Format,
        scene_data_descriptor_layout: vk::DescriptorSetLayout,
    ) -> Self {
        let frag_shader = ShaderModule::new(device.clone(), "shaders/gltf_metal_rough_frag.spv");
        let vert_shader = ShaderModule::new(device.clone(), "shaders/gltf_metal_rough_vert.spv");

        let gpu_push_constants_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: std::mem::size_of::<GPUDrawPushConstants>() as u32,
        };

        let mut layout_builder = DescriptorLayoutBuilder::new();
        layout_builder.add_binding(
            0,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        );
        layout_builder.add_binding(
            1,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );
        layout_builder.add_binding(
            2,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            vk::ShaderStageFlags::FRAGMENT,
        );

        let material_layout =
            layout_builder.build(device.clone(), vk::DescriptorSetLayoutCreateFlags::empty());

        let layouts = [scene_data_descriptor_layout, material_layout.layout()];

        let layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &gpu_push_constants_range,
            ..Default::default()
        };

        let pipeline_layout = device.create_pipeline_layout(&layout_create_info);

        let opaque_pipeline = GraphicsPipelineBuilder::new()
            .set_shaders(&frag_shader, &vert_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .disable_multisampling()
            .enable_depth_test(vk::TRUE, vk::CompareOp::GREATER_OR_EQUAL)
            .set_color_attachment_format(draw_image_format)
            .set_depth_format(depth_image_format)
            .set_layout(pipeline_layout)
            .build_pipeline(device.clone());

        // #TODO: We recreate the same pipeline layout to prevent double frees => should probably
        // fix this by redisigning the GrphicsPipelineObject (with Arc for layout?)
        let pipeline_layout_tr = device.create_pipeline_layout(&layout_create_info);
        let transparent_pipeline = GraphicsPipelineBuilder::new()
            .set_shaders(&frag_shader, &vert_shader)
            .set_input_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .set_polygon_mode(vk::PolygonMode::FILL)
            .set_cull_mode(vk::CullModeFlags::NONE, vk::FrontFace::CLOCKWISE)
            .disable_multisampling()
            .enable_depth_test(vk::FALSE, vk::CompareOp::GREATER_OR_EQUAL)
            .enable_blending_additive()
            .set_color_attachment_format(draw_image_format)
            .set_depth_format(depth_image_format)
            .set_layout(pipeline_layout_tr)
            .build_pipeline(device.clone());

        let writer = DescriptorWriter::new();
        let opaque_pipeline = Arc::new(MaterialPipeline {
            pipeline: opaque_pipeline,
            layout: pipeline_layout,
        });

        let transparent_pipeline = Arc::new(MaterialPipeline {
            pipeline: transparent_pipeline,
            layout: pipeline_layout_tr,
        });
        Self {
            opaque_pipeline,
            transparent_pipeline,
            material_layout,
            writer,
        }
    }

    fn clear_resources(&mut self, device: &Device) {
        todo!()
    }

    pub fn write_material(
        &mut self,
        device: &Device,
        pass: MaterialPass,
        resources: &MaterialResources,
        descriptor_allocator: &mut DescriptorAllocator,
    ) -> MaterialInstance {
        let pipeline = match pass {
            MaterialPass::Transparent => self.transparent_pipeline.clone(),
            _ => self.opaque_pipeline.clone(),
        };

        let material_set = descriptor_allocator.allocate(self.material_layout.layout());

        self.writer.clear();
        self.writer.add_buffer(
            0,
            resources.data_buffer.buffer(),
            std::mem::size_of::<MaterialConstants>() as u64,
            resources.data_buffer_offset,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
        self.writer.add_image(
            1,
            resources.color_image.image_view(),
            resources.color_sampler.sampler(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        self.writer.add_image(
            2,
            resources.metal_rough_image.image_view(),
            resources.metal_rough_sampler.sampler(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        );
        self.writer.update_descriptor_set(device, material_set);

        MaterialInstance {
            pipeline,
            material_set,
            pass_type: pass,
        }
    }
}

pub struct DrawContext {
    opaque_surfaces: Vec<RenderObject>,
}

impl DrawContext {
    pub fn new() -> Self {
        Self {
            opaque_surfaces: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        self.opaque_surfaces.clear();
    }

    pub fn objects(&self) -> &Vec<RenderObject> {
        &self.opaque_surfaces
    }
}

pub struct RenderObject {
    index_count: u32,
    first_index: usize,
    index_buffer: vk::Buffer,
    material: Arc<MaterialInstance>,
    transform: glm::Mat4,
    vertex_buffer_address: vk::DeviceAddress,
}

impl RenderObject {
    pub fn draw(
        &self,
        device: &Device,
        command_buffer: vk::CommandBuffer,
        descriptor_set0: vk::DescriptorSet,
    ) {
        let descriptor_set1 = self.material.material_set;
        self.bind_pipeline(command_buffer);
        device.draw_mesh(command_buffer, descriptor_set0, descriptor_set1, self);
    }

    pub fn bind_pipeline(&self, command_buffer: vk::CommandBuffer) {
        self.material.bind_pipeline(command_buffer);
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.material.pipeline.layout
    }

    pub fn transform(&self) -> glm::Mat4 {
        self.transform
    }

    pub fn vertex_buffer_address(&self) -> vk::DeviceAddress {
        self.vertex_buffer_address
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }

    pub fn first_index(&self) -> usize {
        self.first_index
    }
    pub fn index_buffer(&self) -> vk::Buffer {
        self.index_buffer
    }
}

pub trait Rendereable {
    fn draw(&self, top_matrix: &glm::Mat4, ctx: &mut DrawContext);
}

pub struct MeshNode {
    children: Vec<Arc<Mutex<MeshNode>>>,
    parent: Option<Weak<MeshNode>>,
    local_transform: glm::Mat4,
    world_transform: glm::Mat4,
    mesh: MeshAsset,
}

impl MeshNode {
    pub fn new(mesh: MeshAsset, local_transform: glm::Mat4, world_transform: glm::Mat4) -> Self {
        Self {
            children: Vec::new(),
            parent: None,
            local_transform,
            world_transform,
            mesh,
        }
    }

    pub fn refresh_transform(&mut self, parent_matrix: &glm::Mat4) {
        self.world_transform = parent_matrix * self.local_transform;
        for child in &self.children {
            child
                .lock()
                .expect("Pls no poiserino")
                .refresh_transform(&self.world_transform);
        }
    }

    pub fn name(&self) -> &str {
        self.mesh.name()
    }
}

impl Rendereable for MeshNode {
    fn draw(&self, top_matrix: &glm::Mat4, ctx: &mut DrawContext) {
        let node_matrix = top_matrix * self.world_transform;
        let index_buffer = self.mesh.buffers().index_buffer();
        let vertex_buffer_address = self.mesh.buffers().vertex_buffer_address();
        for surface in self.mesh.surfaces() {
            let render_obj = RenderObject {
                index_count: surface.count(),
                first_index: surface.start_idx(),
                index_buffer,
                material: surface.material().clone(),
                transform: node_matrix,
                vertex_buffer_address,
            };
            ctx.opaque_surfaces.push(render_obj);
        }
        for child in &self.children {
            child
                .lock()
                .expect("Pls no poiserino")
                .draw(top_matrix, ctx);
        }
    }
}
