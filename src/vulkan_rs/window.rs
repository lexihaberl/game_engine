use super::device::Device;
use super::instance::Instance;
use super::utils;
use ash::{
    ext::metal_surface,
    khr::{android_surface, surface, wayland_surface, win32_surface, xcb_surface, xlib_surface},
    vk::{self, SurfaceKHR},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
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

pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    display_handle: RawDisplayHandle,
    window_handle: RawWindowHandle,
    allocation_callbacks: Option<&vk::AllocationCallbacks<'_>>,
) -> SurfaceKHR {
    let surface_opt = match (display_handle, window_handle) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(window)) => {
            let surface_desc = vk::Win32SurfaceCreateInfoKHR::default()
                .hwnd(window.hwnd.get())
                .hinstance(
                    window
                        .hinstance
                        .expect("Win32 hinstance should be available!")
                        .get(),
                );
            let surface_fn = win32_surface::Instance::new(entry, instance);
            surface_fn.create_win32_surface(&surface_desc, allocation_callbacks)
        }

        (RawDisplayHandle::Wayland(display), RawWindowHandle::Wayland(window)) => {
            let surface_desc = vk::WaylandSurfaceCreateInfoKHR::default()
                .display(display.display.as_ptr())
                .surface(window.surface.as_ptr());
            let surface_fn = wayland_surface::Instance::new(entry, instance);
            surface_fn.create_wayland_surface(&surface_desc, allocation_callbacks)
        }

        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let surface_desc = vk::XlibSurfaceCreateInfoKHR::default()
                .dpy(
                    display
                        .display
                        .expect("Xlib display should be available!")
                        .as_ptr(),
                )
                .window(window.window);
            let surface_fn = xlib_surface::Instance::new(entry, instance);
            surface_fn.create_xlib_surface(&surface_desc, allocation_callbacks)
        }

        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let surface_desc = vk::XcbSurfaceCreateInfoKHR::default()
                .connection(
                    display
                        .connection
                        .expect("Xcb connection should be available!")
                        .as_ptr(),
                )
                .window(window.window.get());
            let surface_fn = xcb_surface::Instance::new(entry, instance);
            surface_fn.create_xcb_surface(&surface_desc, allocation_callbacks)
        }

        (RawDisplayHandle::Android(_), RawWindowHandle::AndroidNdk(window)) => {
            let surface_desc =
                vk::AndroidSurfaceCreateInfoKHR::default().window(window.a_native_window.as_ptr());
            let surface_fn = android_surface::Instance::new(entry, instance);
            surface_fn.create_android_surface(&surface_desc, allocation_callbacks)
        }

        #[cfg(target_os = "macos")]
        (RawDisplayHandle::AppKit(_), RawWindowHandle::AppKit(window)) => {
            use raw_window_metal::{appkit, Layer};

            let layer = match appkit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
            let surface_fn = metal_surface::Instance::new(entry, instance);
            surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
        }

        #[cfg(target_os = "ios")]
        (RawDisplayHandle::UiKit(_), RawWindowHandle::UiKit(window)) => {
            use raw_window_metal::{uikit, Layer};

            let layer = match uikit::metal_layer_from_handle(window) {
                Layer::Existing(layer) | Layer::Allocated(layer) => layer.cast(),
            };

            let surface_desc = vk::MetalSurfaceCreateInfoEXT::default().layer(&*layer);
            let surface_fn = metal_surface::Instance::new(entry, instance);
            surface_fn.create_metal_surface(&surface_desc, allocation_callbacks)
        }

        _ => panic!("Unsupported display handle"),
    };
    surface_opt.expect("Device should have enough memory!")
}

pub struct Surface {
    handle: vk::SurfaceKHR,
    loader: ash::khr::surface::Instance,
    instance: Arc<Instance>,
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
        let surface = unsafe {
            create_surface(
                &instance.entry,
                &instance.handle,
                raw_display_handle,
                raw_window_handle,
                None,
            )
        };
        let loader = ash::khr::surface::Instance::new(&instance.entry, &instance.handle);

        Arc::new(Surface {
            handle: surface,
            loader,
            instance,
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

        let swapchain_loader =
            ash::khr::swapchain::Device::new(&self.instance.handle, &device.handle);
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

        let presentation_queue = device.presentation_queue;
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
