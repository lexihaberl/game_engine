use crate::vulkan_rs::debug;
use crate::vulkan_rs::window;
use crate::vulkan_rs::AppInfo;
use crate::vulkan_rs::Device;
use crate::vulkan_rs::EngineInfo;
use crate::vulkan_rs::Instance;
use crate::vulkan_rs::PhysicalDeviceSelector;
use crate::vulkan_rs::Surface;
use crate::vulkan_rs::Swapchain;
use crate::vulkan_rs::Version;
use ash::vk;
use raw_window_handle::HasDisplayHandle;
use std::sync::Arc;
use winit::window::Window;

pub struct FrameData {
    device: Arc<Device>,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    result_presentable_semaphore: vk::Semaphore,
    in_flight_fence: vk::Fence,
}

impl FrameData {
    fn new(device: Arc<Device>) -> FrameData {
        let command_pool = device.create_command_pool();
        let command_buffer = device.create_command_buffer(command_pool);
        let image_available_semaphore = device.create_semaphore();
        let result_presentable_semaphore = device.create_semaphore();
        let in_flight_fence = device.create_fence(vk::FenceCreateFlags::SIGNALED);
        FrameData {
            device,
            command_pool,
            command_buffer,
            image_available_semaphore,
            result_presentable_semaphore,
            in_flight_fence,
        }
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        self.device.destroy_command_pool(self.command_pool);
        self.device
            .destroy_semaphore(self.image_available_semaphore);
        self.device
            .destroy_semaphore(self.result_presentable_semaphore);
        self.device.destroy_fence(self.in_flight_fence);
    }
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanRenderer {
    #[allow(dead_code)]
    instance: Arc<Instance>,
    #[allow(dead_code)]
    debug_messenger: Option<debug::DebugMessenger>,
    #[allow(dead_code)]
    surface: Arc<Surface>,
    #[allow(dead_code)]
    physical_device: vk::PhysicalDevice,
    device: Arc<Device>,
    swapchain: Swapchain,
    frame_data: Vec<FrameData>,
    frame_index: usize,
}

impl VulkanRenderer {
    pub fn new(window: Arc<Window>) -> VulkanRenderer {
        let raw_display_handle = window
            .display_handle()
            .expect("I hope window has a display handle")
            .as_raw();
        let mut required_extensions = window::get_required_instance_extensions(raw_display_handle);
        let (required_layers, debug_messenger_create_info) = if cfg!(debug_assertions) {
            log::info!("Debug mode enabled, enabling validation layers");
            let required_debug_extensions = debug::get_required_extensions();
            required_extensions.extend(required_debug_extensions);
            (
                debug::get_required_layers(),
                Some(debug::DebugMessenger::fill_create_info()),
            )
        } else {
            log::info!("Debug mode disabled, not enabling validation layers");
            (vec![], None)
        };
        log::debug!("Required extensions: {:?}", required_extensions);
        log::debug!("Required layers: {:?}", required_layers);
        let min_vulkan_version = Version {
            major: 1,
            minor: 3,
            patch: 0,
        };
        let app_info = AppInfo {
            name: "Vulkan Renderer".to_string(),
            version: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
        };
        let engine_info = EngineInfo {
            name: "Vulkan Engine".to_string(),
            version: Version {
                major: 1,
                minor: 0,
                patch: 0,
            },
            vulkan_version: min_vulkan_version,
        };
        let instance = Instance::new(
            app_info,
            engine_info,
            &required_layers,
            &required_extensions,
            debug_messenger_create_info,
        );
        let debug_messenger = if cfg!(debug_assertions) {
            log::info!("Creating debug messenger");
            Some(debug::DebugMessenger::new(instance.clone()))
        } else {
            None
        };
        let surface = window::Surface::new(instance.clone(), window.clone());

        let physical_device_selector = PhysicalDeviceSelector::new(min_vulkan_version);
        let physical_device = physical_device_selector.select(instance.clone(), &surface);

        let device = Device::new(instance.clone(), &physical_device, &surface);

        let swapchain = surface.create_swapchain(
            &physical_device,
            device.clone(),
            window.inner_size().to_logical(window.scale_factor()),
        );

        let mut frame_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frame_data.push(FrameData::new(device.clone()));
        }

        VulkanRenderer {
            surface,
            instance,
            debug_messenger,
            physical_device,
            device,
            swapchain,
            frame_data,
            frame_index: 0,
        }
    }

    fn get_current_frame(&self) -> &FrameData {
        &self.frame_data[self.frame_index % MAX_FRAMES_IN_FLIGHT]
    }

    pub fn draw(&mut self) {
        let current_frame = self.get_current_frame();
        // MAX_IN_FLIGHT_FRAMES is 2 => we wait for the frame before the previous one to finish.
        self.device
            .wait_for_fence(&current_frame.in_flight_fence, 1_000_000_000); //1E9 ns -> 1s
        self.device.reset_fence(&current_frame.in_flight_fence);

        let (image_index, image) = self
            .swapchain
            .acquire_next_image(current_frame.image_available_semaphore, 1_000_000_000);

        let command_buffer = current_frame.command_buffer;
        // commands are finished -> can reset command buffer
        self.device.reset_command_buffer(command_buffer);

        // start recording commands
        self.device
            .begin_command_buffer(command_buffer, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.transition_image_layout(
            command_buffer,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );
        let flash_color = (self.frame_index as f32 / 100.0).sin().abs();
        let clear_value = vk::ClearColorValue {
            float32: [0.0, 0.0, flash_color, 1.0],
        };
        self.device.cmd_clear_color_image(
            command_buffer,
            image,
            vk::ImageLayout::GENERAL,
            &clear_value,
        );
        self.device.transition_image_layout(
            command_buffer,
            image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        self.device.end_command_buffer(command_buffer);
        self.submit_to_queue(current_frame, current_frame.in_flight_fence);
        self.swapchain
            .present_image(current_frame.result_presentable_semaphore, image_index);
        self.frame_index += 1;
    }

    fn submit_to_queue(&self, current_frame: &FrameData, fence: vk::Fence) {
        // command_buffer: is the clear cmd buffer
        // when submitting -> we say that this cmd buffer should be executed
        // when the image_available_semaphore was signaled (i.e. the image is available)
        // and after the cmd buffer is executed, the result_presentable_semaphore will be signaled
        // so that we can present the image to the surface
        let cmd_buffer_submit_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: current_frame.command_buffer,
            p_next: std::ptr::null(),
            ..Default::default()
        };
        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: current_frame.image_available_semaphore,
            stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            p_next: std::ptr::null(),
            device_index: 0,
            value: 1,
            ..Default::default()
        };
        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfo {
            s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
            semaphore: current_frame.result_presentable_semaphore,
            stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS,
            p_next: std::ptr::null(),
            device_index: 0,
            value: 1,
            ..Default::default()
        };
        let submit_info = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            p_next: std::ptr::null(),
            wait_semaphore_info_count: 1,
            p_wait_semaphore_infos: &wait_semaphore_submit_info,
            signal_semaphore_info_count: 1,
            p_signal_semaphore_infos: &signal_semaphore_submit_info,
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_buffer_submit_info,
            ..Default::default()
        };
        self.device.submit_to_graphics_queue(submit_info, fence);
    }

    pub fn wait_idle(&self) {
        self.device.wait_idle();
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        self.device.wait_idle();
    }
}
