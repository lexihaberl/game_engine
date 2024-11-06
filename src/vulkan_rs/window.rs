use super::instance::Instance;
use ash::{
    ext::metal_surface,
    khr::{android_surface, surface, wayland_surface, win32_surface, xcb_surface, xlib_surface},
    vk::{self, SurfaceKHR},
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::CString;
use std::sync::Arc;
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
    pub handle: vk::SurfaceKHR,
    pub loader: ash::khr::surface::Instance,
    // instance and window are part of struct since they have to live
    // longer than the surface
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
            _instance: instance,
            _window: window,
        })
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
