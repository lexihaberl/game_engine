use crate::vulkan_rs::Device;
use ash::vk;
use gpu_allocator::vulkan::Allocation;
use gpu_allocator::vulkan::AllocationCreateDesc;
use gpu_allocator::vulkan::AllocationScheme;
use std::sync::Arc;
use std::sync::Mutex;

pub struct Allocator {
    #[allow(dead_code)]
    device: Arc<Device>,
    allocator: gpu_allocator::vulkan::Allocator,
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
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        };
        let allocation = self
            .allocator
            .allocate(&allocation_create_desc)
            .expect("I pray that this never fails");
        self.device
            .bind_image_memory(image, unsafe { allocation.memory() }, allocation.offset());
        allocation
    }

    pub fn free_allocation(&mut self, allocation: Allocation) {
        self.allocator
            .free(allocation)
            .expect("I pray that this never fails");
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
    ) -> Self {
        let image = device.create_image(format, usage_flags, extent);
        let image_mem_req = device.get_image_memory_requirements(image);

        let allocation = allocator
            .lock()
            .expect("Mutex has been poisoned and i dont wanan handle it yet")
            .allocate_image(image, image_mem_req);
        let image_view = device.create_image_view(image, format, aspect_flags);
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
