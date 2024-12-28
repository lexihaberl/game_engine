use super::ImmediateCommandData;
use crate::vulkan_rs::Device;
use ash::vk;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;
use std::sync::Arc;
use std::sync::Mutex;

pub struct Allocator {
    // NOTE: allocator has to be dropped before device to ensure that the device
    // is still alive when the allocator is dropped.
    allocator: gpu_allocator::vulkan::Allocator,
    #[allow(dead_code)]
    device: Arc<Device>,
}

impl Allocator {
    pub fn new(device: Arc<Device>) -> Arc<Mutex<Self>> {
        let allocator = device.create_allocator();

        Arc::new(Mutex::new(Self { device, allocator }))
    }

    pub fn allocate_image(
        &mut self,
        image: vk::Image,
        image_memory_req: vk::MemoryRequirements,
    ) -> Allocation {
        let allocation_create_desc = AllocationCreateDesc {
            name: "Image",
            location: gpu_allocator::MemoryLocation::GpuOnly,
            requirements: image_memory_req,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .allocate(&allocation_create_desc)
            .expect("I pray that this never fails");
        self.device
            .bind_image_memory(image, unsafe { allocation.memory() }, allocation.offset());
        allocation
    }

    pub fn allocate_buffer(
        &mut self,
        buffer_name: &str,
        buffer: vk::Buffer,
        buffer_memory_req: vk::MemoryRequirements,
        location: gpu_allocator::MemoryLocation,
    ) -> Allocation {
        let allocation_create_desc = AllocationCreateDesc {
            name: buffer_name,
            requirements: buffer_memory_req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .allocate(&allocation_create_desc)
            .expect("I pray that this never fails");
        self.device
            .bind_buffer_memory(buffer, unsafe { allocation.memory() }, allocation.offset());
        allocation
    }

    pub fn free_allocation(&mut self, allocation: Allocation) {
        log::debug!("Freeing allocation");
        self.allocator
            .free(allocation)
            .expect("I pray that this never fails");
    }
}

impl Drop for Allocator {
    fn drop(&mut self) {
        log::debug!("Dropping allocator");
    }
}

pub struct AllocatedImage {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    image: vk::Image,
    image_view: vk::ImageView,
    allocation: Option<Allocation>,
    extent: vk::Extent3D,
    format: vk::Format,
}

impl AllocatedImage {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Self {
        let image = device.create_image(format, usage_flags, extent, mip_levels);
        let image_mem_req = device.get_image_memory_requirements(image);

        let allocation = allocator
            .lock()
            .expect("Mutex has been poisoned and i dont wanan handle it yet")
            .allocate_image(image, image_mem_req);
        let image_view = device.create_image_view(image, format, aspect_flags, mip_levels);
        Self {
            device,
            allocator,
            image,
            image_view,
            allocation: Some(allocation),
            extent,
            format,
        }
    }

    pub fn new_draw_color_image(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        extent: vk::Extent3D,
    ) -> Self {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        let format = vk::Format::R16G16B16A16_SFLOAT;
        let aspect = vk::ImageAspectFlags::COLOR;
        Self::new(device, allocator, format, usage, extent, aspect, 1)
    }

    pub fn new_depth_image(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        extent: vk::Extent3D,
    ) -> Self {
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        let format = vk::Format::D32_SFLOAT;
        let aspect_flags = vk::ImageAspectFlags::DEPTH;
        Self::new(device, allocator, format, usage, extent, aspect_flags, 1)
    }

    fn allocate_texture(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        mip_mapped: bool,
    ) -> Self {
        let mip_levels = if mip_mapped {
            f32::floor(f32::log2(u32::max(extent.width, extent.height) as f32)) as u32 + 1
        } else {
            1
        };
        let aspect_flags = if format == vk::Format::D32_SFLOAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        Self::new(
            device,
            allocator,
            format,
            usage_flags,
            extent,
            aspect_flags,
            mip_levels,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_texture<T: Copy>(
        data: &[T],
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        format: vk::Format,
        usage_flags: vk::ImageUsageFlags,
        extent: vk::Extent3D,
        mip_mapped: bool,
        immediate_command: &ImmediateCommandData,
    ) -> Self {
        let size = extent.width * extent.height * extent.depth * 4;
        let mut staging_buffer = AllocatedBuffer::new(
            device.clone(),
            allocator.clone(),
            "Texture Staging Buffer",
            vk::BufferUsageFlags::TRANSFER_SRC,
            size as u64,
            gpu_allocator::MemoryLocation::CpuToGpu,
        );
        staging_buffer.copy_from_slice(data, 0);

        let image = Self::allocate_texture(
            device.clone(),
            allocator.clone(),
            format,
            usage_flags | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            mip_mapped,
        );
        immediate_command.immediate_submit(|device, cmd| {
            let image = image.image();
            device.transition_image_layout(
                cmd,
                image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
            let copy_region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: extent,
            };
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer.buffer(),
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
            device.transition_image_layout(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
        });
        image
    }

    pub fn image(&self) -> vk::Image {
        self.image
    }

    pub fn extent(&self) -> vk::Extent3D {
        self.extent
    }

    pub fn image_view(&self) -> vk::ImageView {
        self.image_view
    }
    pub fn format(&self) -> vk::Format {
        self.format
    }
}

impl Drop for AllocatedImage {
    fn drop(&mut self) {
        log::debug!("Dropping allocated image");
        self.device.destroy_image_view(self.image_view);
        self.allocator
            .lock()
            .expect("Mutex has been poisoned and i dont wanan handle it yet")
            .free_allocation(
                self.allocation
                    .take()
                    .expect("Allocation should exist until its dropped"),
            );
        self.device.destroy_image(self.image);
    }
}

pub struct AllocatedBuffer {
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
    cpu_accesible: bool,
}

impl AllocatedBuffer {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        buffer_name: &str,
        usage: vk::BufferUsageFlags,
        size: vk::DeviceSize,
        location: gpu_allocator::MemoryLocation,
    ) -> Self {
        let buffer = device.create_buffer(usage, size);
        let mem_requirements = device.get_buffer_memory_requirements(buffer);
        let allocation = allocator
            .lock()
            .expect("Mutex has been poisoned and i dont wanan handle it yet")
            .allocate_buffer(buffer_name, buffer, mem_requirements, location);
        let cpu_accesible = location == gpu_allocator::MemoryLocation::CpuToGpu;
        Self {
            device,
            allocator,
            buffer,
            allocation: Some(allocation),
            cpu_accesible,
        }
    }

    pub fn get_device_address(&self) -> vk::DeviceAddress {
        self.device.get_buffer_device_address(self.buffer)
    }

    pub fn copy_from_slice<T: Copy>(&mut self, data: &[T], offset: usize) {
        if !self.cpu_accesible {
            panic!("Cannot copy to buffer that is not cpu accesible");
        }
        if let Some(allocation) = &mut self.allocation {
            //TODO: maybe add some alignment stuff? refer to gpu allocator crate
            let copy_record = presser::copy_from_slice_to_offset(data, allocation, offset)
                .expect("I pray that this never fails");
            assert!(copy_record.copy_start_offset == offset);
        }
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
}

impl Drop for AllocatedBuffer {
    fn drop(&mut self) {
        log::debug!("Dropping allocated buffer");
        self.allocator
            .lock()
            .expect("Mutex has been poisoned and i dont wanan handle it yet")
            .free_allocation(
                self.allocation
                    .take()
                    .expect("Allocation should exist until its dropped"),
            );
        self.device.destroy_buffer(self.buffer);
    }
}
