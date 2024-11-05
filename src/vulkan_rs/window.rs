use ash::khr::{surface, wayland_surface};
use std::ffi::CString;

pub fn get_required_instance_extensions() -> Vec<CString> {
    let wayland_extensions = vec![wayland_surface::NAME.to_owned(), surface::NAME.to_owned()];
    wayland_extensions
}
