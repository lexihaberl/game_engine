use super::device::Device;
use super::shader::ShaderModule;
use ash::vk;
use std::sync::Arc;

pub struct Pipeline {
    device: Arc<Device>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl Pipeline {
    pub fn create_compute_pipeline(
        device: Arc<Device>,
        set_layouts: &[vk::DescriptorSetLayout],
        shader: ShaderModule,
    ) -> Self {
        let layout_create_info = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };
        let pipeline_layout = device.create_pipeline_layout(&layout_create_info);
        let stage_info = vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: std::ptr::null(),
            stage: vk::ShaderStageFlags::COMPUTE,
            module: shader.module(),
            p_name: b"main\0".as_ptr() as *const i8,
            ..Default::default()
        };

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

        self.device.execute_compute_pipeline(
            command_buffer,
            self.pipeline,
            self.pipeline_layout,
            descriptor_sets,
            group_counts,
        )
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        log::debug!("Dropping pipeline");
        self.device.destroy_pipeline(self.pipeline);
        self.device.destroy_pipeline_layout(self.pipeline_layout);
    }
}
