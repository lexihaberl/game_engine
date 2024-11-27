use super::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct ImmediateCommandData {
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl ImmediateCommandData {
    pub fn new(device: Arc<Device>) -> Self {
        let command_pool = device.create_command_pool();
        let command_buffer = device.create_command_buffer(command_pool);
        let fence = device.create_fence(vk::FenceCreateFlags::SIGNALED);
        Self {
            device,
            command_pool,
            command_buffer,
            fence,
        }
    }

    pub fn immediate_submit<F>(&self, commands: F)
    where
        F: FnOnce(&Device, vk::CommandBuffer),
    {
        self.device.reset_fence(&self.fence);
        self.device.reset_command_buffer(self.command_buffer);
        self.device.begin_command_buffer(
            self.command_buffer,
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        );
        commands(&self.device, self.command_buffer);
        self.device.end_command_buffer(self.command_buffer);
        let submit_info = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            p_next: std::ptr::null(),
            command_buffer_info_count: 1,
            p_command_buffer_infos: &vk::CommandBufferSubmitInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
                p_next: std::ptr::null(),
                command_buffer: self.command_buffer,
                ..Default::default()
            },
            ..Default::default()
        };
        self.device
            .submit_to_graphics_queue(submit_info, self.fence);
        self.device.wait_for_fence(&self.fence, u64::MAX);
    }
}

impl Drop for ImmediateCommandData {
    fn drop(&mut self) {
        log::debug!("Dropping ImmediateCommandData");
        self.device.destroy_command_pool(self.command_pool);
        self.device.destroy_fence(self.fence);
    }
}
