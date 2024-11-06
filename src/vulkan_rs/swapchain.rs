use super::device::Device;
use super::instance::Instance;
use super::utils;
use super::window::Surface;
use ash::vk;
use std::sync::Arc;
use winit::dpi::LogicalSize;

#[derive(Debug)]
#[allow(dead_code)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    pub fn query_support_details(
        surface: &Surface,
        device: &vk::PhysicalDevice,
    ) -> SwapChainSupportDetails {
        let surface_instance = &surface.loader;
        let surface = surface.handle;
        let capabilities = unsafe {
            surface_instance
                .get_physical_device_surface_capabilities(*device, surface)
                .expect("Could not get surface capabilities")
        };
        let surface_formats = unsafe {
            surface_instance
                .get_physical_device_surface_formats(*device, surface)
                .expect("Could not get surface formats")
        };
        let present_modes = unsafe {
            surface_instance
                .get_physical_device_surface_present_modes(*device, surface)
                .expect("Could not get present modes")
        };
        SwapChainSupportDetails {
            capabilities,
            surface_formats,
            present_modes,
        }
    }
}

pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    pub swapchain: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
}

impl Swapchain {
    pub fn new(
        instance: Arc<Instance>,
        surface: Arc<Surface>,
        physical_device: &vk::PhysicalDevice,
        device: Arc<Device>,
        window_size: LogicalSize<u32>,
    ) -> Self {
        let support_details =
            SwapChainSupportDetails::query_support_details(&surface, physical_device);

        let surface_format = Self::choose_swap_surface_format(&support_details.surface_formats);
        let present_mode = Self::choose_swap_present_mode(&support_details.present_modes);
        let extent = Self::choose_swap_extent(&support_details.capabilities, window_size);

        let mut image_count = support_details.capabilities.min_image_count + 1;
        if support_details.capabilities.max_image_count > 0 {
            image_count = image_count.min(support_details.capabilities.max_image_count);
        }

        let indices_array = [
            device.graphics_queue_family_idx,
            device.presentation_queue_family_idx,
        ];
        let (image_sharing_mode, queue_fam_index_count, p_queue_fam_indices) =
            if device.graphics_queue_family_idx != device.presentation_queue_family_idx {
                (vk::SharingMode::CONCURRENT, 2, indices_array.as_ptr())
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, std::ptr::null())
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: surface.handle,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            queue_family_index_count: queue_fam_index_count,
            p_queue_family_indices: p_queue_fam_indices,
            pre_transform: support_details.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            p_next: std::ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            ..Default::default()
        };

        let swapchain_loader = ash::khr::swapchain::Device::new(&instance.handle, &device.handle);
        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("Could not create swapchain")
        };
        let swapchain_images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Device should not be out of memory")
        };
        let image_views =
            Self::create_image_views(&device.handle, surface_format.format, &swapchain_images);

        Self {
            device,
            surface,
            swapchain,
            swapchain_loader,
            images: swapchain_images,
            image_views,
            extent,
            format: surface_format.format,
        }
    }

    fn choose_swap_surface_format(
        available_formats: &[vk::SurfaceFormatKHR],
    ) -> vk::SurfaceFormatKHR {
        let desired_format = available_formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });
        match desired_format {
            Some(format) => *format,
            None => *available_formats.first().expect(
                "Should not be empty, since we checked for the existence of atleast one format",
            ),
        }
    }

    fn choose_swap_present_mode(
        available_present_modes: &[vk::PresentModeKHR],
    ) -> vk::PresentModeKHR {
        let desired_mode = available_present_modes
            .iter()
            .find(|mode| **mode == vk::PresentModeKHR::MAILBOX);
        match desired_mode {
            Some(mode) => *mode,
            // FIFO is guaranteed to be available
            None => vk::PresentModeKHR::FIFO,
        }
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window_size: LogicalSize<u32>,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: utils::clamp(
                    window_size.width,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: utils::clamp(
                    window_size.height,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }
    fn create_image_views(
        device: &ash::Device,
        format: vk::Format,
        swapchain_images: &[vk::Image],
    ) -> Vec<vk::ImageView> {
        let mut swapchain_views: Vec<vk::ImageView> = Vec::with_capacity(swapchain_images.len());
        for image in swapchain_images.iter() {
            let create_info = vk::ImageViewCreateInfo {
                s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
                image: *image,
                view_type: vk::ImageViewType::TYPE_2D,
                format,
                components: vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                },
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                p_next: std::ptr::null(),
                flags: vk::ImageViewCreateFlags::empty(),
                ..Default::default()
            };
            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .expect("Device hopefully not out of memory")
            };
            swapchain_views.push(image_view);
        }
        swapchain_views
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.image_views.iter() {
                self.device.handle.destroy_image_view(*image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
