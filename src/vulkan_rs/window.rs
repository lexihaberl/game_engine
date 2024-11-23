use super::device::Device;
use super::instance::Instance;
use super::utils;
use ash::{
    ext::metal_surface,
    khr::{android_surface, surface, wayland_surface, win32_surface, xcb_surface, xlib_surface},
    vk,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use std::ffi::CString;
use std::sync::Arc;
use winit::dpi::LogicalSize;
use winit::window::Window;

pub fn get_required_instance_extensions(display_handle: RawDisplayHandle) -> Vec<CString> {
    match display_handle {
        RawDisplayHandle::Windows(_) => {
            vec![win32_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        RawDisplayHandle::Wayland(_) => {
            vec![wayland_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        RawDisplayHandle::Xlib(_) => {
            vec![xlib_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        RawDisplayHandle::Xcb(_) => {
            vec![xcb_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        RawDisplayHandle::Android(_) => {
            vec![android_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        RawDisplayHandle::AppKit(_) | RawDisplayHandle::UiKit(_) => {
            vec![metal_surface::NAME.to_owned(), surface::NAME.to_owned()]
        }

        _ => panic!("Unsupported display handle"),
    }
}

pub struct Surface {
    handle: vk::SurfaceKHR,
    loader: ash::khr::surface::Instance,
    _instance: Arc<Instance>,
    _window: Arc<Window>,
}

impl Surface {
    pub fn new(instance: Arc<Instance>, window: Arc<Window>) -> Arc<Surface> {
        let raw_window_handle = window
            .window_handle()
            .expect("I hope the window handle exists")
            .as_raw();
        let raw_display_handle = window
            .display_handle()
            .expect("I hope window has a display handle")
            .as_raw();
        let surface = instance.create_surface(raw_display_handle, raw_window_handle, None);
        let loader = instance.create_surface_loader();

        Arc::new(Surface {
            handle: surface,
            loader,
            _instance: instance,
            _window: window,
        })
    }

    pub fn get_physical_device_surface_support(
        &self,
        device: &vk::PhysicalDevice,
        idx: u32,
    ) -> bool {
        unsafe {
            self.loader
                .get_physical_device_surface_support(*device, idx, self.handle)
                .expect("Host does not have enough resources or smth")
        }
    }

    pub fn query_support_details(&self, device: &vk::PhysicalDevice) -> SwapChainSupportDetails {
        let surface_instance = &self.loader;
        let surface = self.handle;
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

    pub fn create_swapchain(
        self: &Arc<Self>,
        physical_device: &vk::PhysicalDevice,
        device: Arc<Device>,
        window_size: LogicalSize<u32>,
    ) -> Swapchain {
        let support_details = self.query_support_details(physical_device);

        let surface_format = Self::choose_swap_surface_format(&support_details.surface_formats);
        let present_mode = Self::choose_swap_present_mode(&support_details.present_modes);
        let extent = Self::choose_swap_extent(&support_details.capabilities, window_size);

        let mut image_count = support_details.capabilities.min_image_count + 1;
        if support_details.capabilities.max_image_count > 0 {
            image_count = image_count.min(support_details.capabilities.max_image_count);
        }

        let graphics_queue_family_idx = device.get_graphics_queue_idx();
        let presentation_queue_family_idx = device.get_presentation_queue_idx();

        let indices_array = [graphics_queue_family_idx, presentation_queue_family_idx];
        let (image_sharing_mode, queue_fam_index_count, p_queue_fam_indices) =
            if graphics_queue_family_idx != presentation_queue_family_idx {
                (vk::SharingMode::CONCURRENT, 2, indices_array.as_ptr())
            } else {
                (vk::SharingMode::EXCLUSIVE, 0, std::ptr::null())
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            surface: self.handle,
            min_image_count: image_count,
            image_format: surface_format.format,
            image_color_space: surface_format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
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

        let swapchain_loader = device.create_swapchain_loader();
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
        let image_views = device.create_image_views(surface_format.format, &swapchain_images);

        let presentation_queue = device.get_presentation_queue();

        Swapchain {
            device,
            surface: self.clone(),
            swapchain,
            swapchain_loader,
            images: swapchain_images,
            image_views,
            extent,
            presentation_queue,
            format: surface_format.format,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        log::debug!("Destroying surface!");
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct SwapChainSupportDetails {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct Swapchain {
    device: Arc<Device>,
    surface: Arc<Surface>,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: ash::khr::swapchain::Device,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    extent: vk::Extent2D,
    format: vk::Format,
    presentation_queue: vk::Queue,
}

impl Swapchain {
    pub fn acquire_next_image(&self, semaphore: vk::Semaphore, timeout: u64) -> (u32, vk::Image) {
        let result = unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                timeout,
                semaphore,
                vk::Fence::null(),
            )
        };
        match result {
            Ok((image_index, _is_surface_suboptimal)) => {
                (image_index, self.images[image_index as usize])
            }
            Err(e) => panic!("Failed to acquire next image: {:?}", e),
        }
    }

    pub fn present_image(&self, wait_semaphore: vk::Semaphore, image_index: u32) {
        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: std::ptr::null(),
            swapchain_count: 1,
            p_swapchains: &self.swapchain,
            p_wait_semaphores: &wait_semaphore,
            wait_semaphore_count: 1,
            p_image_indices: &image_index,
            ..Default::default()
        };

        unsafe {
            self.swapchain_loader
                .queue_present(self.presentation_queue, &present_info)
                .expect("Failed to present image");
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for image_view in self.image_views.iter() {
                self.device.destroy_image_view(*image_view);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
