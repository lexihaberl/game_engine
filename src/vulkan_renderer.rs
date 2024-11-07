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
}

impl FrameData {
    fn new(device: Arc<Device>) -> FrameData {
        let command_pool = device.create_command_pool();
        let command_buffer = device.create_command_buffer(command_pool);
        FrameData {
            device,
            command_pool,
            command_buffer,
        }
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        self.device.destroy_command_pool(self.command_pool);
    }
}

pub const MAX_FRAMES_IN_FLIGHT: usize = 3;

pub struct VulkanRenderer {
    instance: Arc<Instance>,
    #[allow(dead_code)]
    debug_messenger: Option<debug::DebugMessenger>,
    surface: Arc<Surface>,
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

        let swapchain = Swapchain::new(
            instance.clone(),
            surface.clone(),
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
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        self.device.wait_idle();
    }
}
