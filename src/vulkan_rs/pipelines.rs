use super::device::Device;
use super::shader::ShaderModule;
use super::MeshAsset;
use ash::vk;
use nalgebra_glm::Vec4;
use std::sync::Arc;

#[repr(C)]
#[derive(bytemuck::NoUninit, Copy, Clone, Debug)]
pub struct PushConstants {
    data1: Vec4,
    data2: Vec4,
    data3: Vec4,
    data4: Vec4,
}

impl PushConstants {
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub struct ComputePipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub fn new(
        device: Arc<Device>,
        set_layouts: &[vk::DescriptorSetLayout],
        shader: ShaderModule,
    ) -> Self {
        let push_constants = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<PushConstants>() as u32,
        };
        let layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_constants,
            ..Default::default()
        };
        let pipeline_layout = device.create_pipeline_layout(&layout_create_info);
        let stage_info = shader.create_shader_stage_info(vk::ShaderStageFlags::COMPUTE);

        let pipeline_create_info = vk::ComputePipelineCreateInfo {
            s_type: vk::StructureType::COMPUTE_PIPELINE_CREATE_INFO,
            p_next: std::ptr::null(),
            layout: pipeline_layout,
            stage: stage_info,
            ..Default::default()
        };

        // we pass only one create info => should get exactly one pipeline
        let pipeline = device.create_compute_pipelines(&[pipeline_create_info])[0];
        Self {
            device,
            pipeline,
            pipeline_layout,
        }
    }

    pub fn execute_compute(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
        extent: vk::Extent2D,
    ) {
        let group_counts = [
            (extent.width as f32 / 16.0).ceil() as u32,
            (extent.height as f32 / 16.0).ceil() as u32,
            1,
        ];
        let push_constants = PushConstants {
            data1: Vec4::new(1.0, 0.0, 0.0, 1.0),
            data2: Vec4::new(0.0, 0.0, 1.0, 1.0),
            data3: Vec4::new(0.0, 0.0, 0.0, 0.0),
            data4: Vec4::new(0.0, 0.0, 0.0, 0.0),
        };

        self.device.execute_compute_pipeline(
            command_buffer,
            self.pipeline,
            self.pipeline_layout,
            descriptor_sets,
            group_counts,
            &push_constants,
        )
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        log::debug!("Dropping pipeline");
        self.device.destroy_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
    }
}

pub struct GraphicsPipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub fn begin_drawing(
        &self,
        command_buffer: vk::CommandBuffer,
        color_image: vk::ImageView,
        depth_image: vk::ImageView,
        color_image_layout: vk::ImageLayout,
        depth_image_layout: vk::ImageLayout,
        render_extent: vk::Extent2D,
        clear_color: Option<vk::ClearColorValue>,
    ) {
        let color_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: std::ptr::null(),
            image_view: color_image,
            image_layout: color_image_layout,
            load_op: if clear_color.is_some() {
                vk::AttachmentLoadOp::CLEAR
            } else {
                vk::AttachmentLoadOp::LOAD
            },
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: if let Some(clear_color) = clear_color {
                vk::ClearValue { color: clear_color }
            } else {
                vk::ClearValue::default()
            },
            ..Default::default()
        };

        let depth_attachment_info = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            p_next: std::ptr::null(),
            image_view: depth_image,
            image_layout: depth_image_layout,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
            ..Default::default()
        };

        let rendering_info = vk::RenderingInfo {
            s_type: vk::StructureType::RENDERING_INFO,
            p_next: std::ptr::null(),
            render_area: vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: render_extent,
            },
            layer_count: 1,
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_info,
            p_depth_attachment: &depth_attachment_info,
            p_stencil_attachment: std::ptr::null(),
            ..Default::default()
        };

        let view_port = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: render_extent.width as f32,
            height: render_extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: render_extent,
        };

        self.device.begin_rendering(
            command_buffer,
            &rendering_info,
            self.pipeline,
            view_port,
            scissor,
        )
    }

    pub fn end_drawing(&self, command_buffer: vk::CommandBuffer) {
        self.device.end_rendering(command_buffer);
    }

    pub fn draw(
        &self,
        command_buffer: vk::CommandBuffer,
        render_extent: vk::Extent2D,
        mesh: &MeshAsset,
    ) {
        self.device
            .draw_mesh(command_buffer, self.pipeline_layout, render_extent, mesh);
    }
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        log::debug!("Dropping GraphicsPipeline");
        self.device.destroy_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
    }
}

pub struct GraphicsPipelineBuilder<'a> {
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo<'a>>,
    input_assembly_info: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    rasterizer_info: vk::PipelineRasterizationStateCreateInfo<'a>,
    color_blend_attachment: vk::PipelineColorBlendAttachmentState,
    multisampling_info: vk::PipelineMultisampleStateCreateInfo<'a>,
    depth_stencil_info: vk::PipelineDepthStencilStateCreateInfo<'a>,
    rendering_info: vk::PipelineRenderingCreateInfo<'a>,
    color_attachment_format: vk::Format,
    pipeline_layout: Option<vk::PipelineLayout>,
}

#[allow(dead_code)]
impl<'a> GraphicsPipelineBuilder<'a> {
    pub fn new() -> Self {
        Self {
            shader_stages: Vec::new(),
            input_assembly_info: vk::PipelineInputAssemblyStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                ..Default::default()
            },
            rasterizer_info: vk::PipelineRasterizationStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                ..Default::default()
            },
            color_blend_attachment: vk::PipelineColorBlendAttachmentState {
                ..Default::default()
            },
            multisampling_info: vk::PipelineMultisampleStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                ..Default::default()
            },
            depth_stencil_info: vk::PipelineDepthStencilStateCreateInfo {
                s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                ..Default::default()
            },
            rendering_info: vk::PipelineRenderingCreateInfo {
                s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
                ..Default::default()
            },
            color_attachment_format: vk::Format::UNDEFINED,
            pipeline_layout: None,
        }
    }

    pub fn build_pipeline(mut self, device: Arc<Device>) -> GraphicsPipeline {
        //TODO: support multiviewport stuff at some point
        // dont need to set more stuff since we do dynamic viewport
        let viewport_info = vk::PipelineViewportStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };
        //TODO: play around with blending
        let blending_info = vk::PipelineColorBlendStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            logic_op: vk::LogicOp::COPY,
            logic_op_enable: vk::FALSE,
            attachment_count: 1,
            p_attachments: &self.color_blend_attachment,
            ..Default::default()
        };
        // dont need vertex input info since we do vertex pulling
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            ..Default::default()
        };
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo {
            s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            p_next: std::ptr::null(),
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = self.pipeline_layout.take();

        match pipeline_layout {
            Some(pipeline_layout) => {
                let pipeline_info = vk::GraphicsPipelineCreateInfo {
                    s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
                    p_next: &self.rendering_info as *const vk::PipelineRenderingCreateInfo
                        as *const std::ffi::c_void,
                    stage_count: self.shader_stages.len() as u32,
                    p_stages: self.shader_stages.as_ptr(),
                    p_vertex_input_state: &vertex_input_info,
                    p_input_assembly_state: &self.input_assembly_info,
                    p_viewport_state: &viewport_info,
                    p_rasterization_state: &self.rasterizer_info,
                    p_multisample_state: &self.multisampling_info,
                    p_color_blend_state: &blending_info,
                    p_depth_stencil_state: &self.depth_stencil_info,
                    layout: pipeline_layout,
                    p_dynamic_state: &dynamic_state_info,
                    ..Default::default()
                };
                // should return exactly one pipeline since we only pass one create info
                let pipeline = device.create_graphics_pipeline(&[pipeline_info])[0];
                GraphicsPipeline {
                    device,
                    pipeline,
                    pipeline_layout,
                }
            }
            None => panic!("Pipeline layout not set"),
        }
    }

    pub fn set_layout(mut self, layout: vk::PipelineLayout) -> Self {
        self.pipeline_layout = Some(layout);
        self
    }

    pub fn set_shaders(
        mut self,
        fragment_shader: &'a ShaderModule,
        vertex_shader: &'a ShaderModule,
    ) -> Self {
        self.shader_stages
            .push(fragment_shader.create_shader_stage_info(vk::ShaderStageFlags::FRAGMENT));
        self.shader_stages
            .push(vertex_shader.create_shader_stage_info(vk::ShaderStageFlags::VERTEX));
        self
    }

    pub fn set_input_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
        self.input_assembly_info.topology = topology;
        // wont be using primitive restarts
        self.input_assembly_info.primitive_restart_enable = vk::FALSE;
        self
    }

    pub fn set_polygon_mode(mut self, mode: vk::PolygonMode) -> Self {
        self.rasterizer_info.polygon_mode = mode;
        self.rasterizer_info.line_width = 1.0;
        self
    }

    pub fn set_cull_mode(mut self, mode: vk::CullModeFlags, front_face: vk::FrontFace) -> Self {
        self.rasterizer_info.cull_mode = mode;
        self.rasterizer_info.front_face = front_face;
        self
    }

    pub fn disable_multisampling(mut self) -> Self {
        self.multisampling_info.sample_shading_enable = vk::FALSE;
        // 1 sample per pixel => :sparkles: disabled :sparkles:
        self.multisampling_info.rasterization_samples = vk::SampleCountFlags::TYPE_1;
        self.multisampling_info.min_sample_shading = 1.0;
        self.multisampling_info.p_sample_mask = std::ptr::null();
        self.multisampling_info.alpha_to_coverage_enable = vk::FALSE;
        self.multisampling_info.alpha_to_one_enable = vk::FALSE;
        self
    }

    pub fn disable_blending(mut self) -> Self {
        self.color_blend_attachment.blend_enable = vk::FALSE;
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        self
    }

    pub fn set_color_attachment_format(mut self, format: vk::Format) -> Self {
        self.color_attachment_format = format;
        self.rendering_info.p_color_attachment_formats = &self.color_attachment_format;
        self.rendering_info.color_attachment_count = 1;
        self
    }

    pub fn set_depth_format(mut self, format: vk::Format) -> Self {
        self.rendering_info.depth_attachment_format = format;
        self
    }

    pub fn disable_depth_test(mut self) -> Self {
        self.depth_stencil_info.depth_test_enable = vk::FALSE;
        self.depth_stencil_info.depth_write_enable = vk::FALSE;
        self.depth_stencil_info.depth_compare_op = vk::CompareOp::NEVER;
        self.depth_stencil_info.depth_bounds_test_enable = vk::FALSE;
        self.depth_stencil_info.stencil_test_enable = vk::FALSE;
        self.depth_stencil_info.front = vk::StencilOpState::default();
        self.depth_stencil_info.back = vk::StencilOpState::default();
        self.depth_stencil_info.min_depth_bounds = 0.0;
        self.depth_stencil_info.max_depth_bounds = 1.0;
        self
    }

    pub fn enable_depth_test(
        mut self,
        depth_write_enable: vk::Bool32,
        compare_op: vk::CompareOp,
    ) -> Self {
        self.depth_stencil_info.depth_test_enable = vk::TRUE;
        self.depth_stencil_info.depth_write_enable = depth_write_enable;
        self.depth_stencil_info.depth_compare_op = compare_op;
        self.depth_stencil_info.depth_bounds_test_enable = vk::FALSE;
        self.depth_stencil_info.stencil_test_enable = vk::FALSE;
        self.depth_stencil_info.front = vk::StencilOpState::default();
        self.depth_stencil_info.back = vk::StencilOpState::default();
        self.depth_stencil_info.min_depth_bounds = 0.0;
        self.depth_stencil_info.max_depth_bounds = 1.0;
        self
    }

    pub fn enable_blending_additive(mut self) -> Self {
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        self.color_blend_attachment.blend_enable = vk::TRUE;
        self.color_blend_attachment.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
        self.color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
        self.color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
        self.color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;
        self
    }

    pub fn enable_blending_alphablend(mut self) -> Self {
        self.color_blend_attachment.color_write_mask = vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A;
        self.color_blend_attachment.blend_enable = vk::TRUE;
        self.color_blend_attachment.src_color_blend_factor = vk::BlendFactor::SRC_ALPHA;
        self.color_blend_attachment.dst_color_blend_factor = vk::BlendFactor::ONE_MINUS_SRC_ALPHA;
        self.color_blend_attachment.color_blend_op = vk::BlendOp::ADD;
        self.color_blend_attachment.src_alpha_blend_factor = vk::BlendFactor::ONE;
        self.color_blend_attachment.dst_alpha_blend_factor = vk::BlendFactor::ZERO;
        self.color_blend_attachment.alpha_blend_op = vk::BlendOp::ADD;
        self
    }
}

impl<'a> Drop for GraphicsPipelineBuilder<'a> {
    fn drop(&mut self) {
        log::debug!("Dropping GraphicsPipelineBuilder");
        //TODO: handle this better by refactoring other code (probably should write wrappers for
        //all vk structures myself with Drop stuff)
        if self.pipeline_layout.is_some() {
            panic!("layout was not consumed. Did you call the GraphicsPipelineBuilder::build_pipeline() method?");
        }
    }
}
