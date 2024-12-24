use super::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct DescriptorLayoutBuilder<'a> {
    bindings: Vec<vk::DescriptorSetLayoutBinding<'a>>,
}

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub fn new(device: Arc<Device>, layout: vk::DescriptorSetLayout) -> Self {
        Self { device, layout }
    }
    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        log::debug!("Destroying descriptor set layout");
        self.device.destroy_descriptor_set_layout(self.layout);
    }
}

impl<'a> DescriptorLayoutBuilder<'a> {
    pub fn new() -> DescriptorLayoutBuilder<'a> {
        DescriptorLayoutBuilder {
            bindings: Vec::new(),
        }
    }

    pub fn add_binding(
        &mut self,
        binding_idx: u32,
        descriptor_type: vk::DescriptorType,
        stage_flags: vk::ShaderStageFlags,
    ) {
        let binding = vk::DescriptorSetLayoutBinding {
            binding: binding_idx,
            descriptor_type,
            descriptor_count: 1,
            stage_flags,
            ..Default::default()
        };
        self.bindings.push(binding);
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.bindings.clear();
    }

    pub fn build(
        &self,
        device: Arc<Device>,
        flags: vk::DescriptorSetLayoutCreateFlags,
    ) -> DescriptorSetLayout {
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            p_next: std::ptr::null(),
            p_bindings: self.bindings.as_ptr(),
            binding_count: self.bindings.len() as u32,
            flags,
            ..Default::default()
        };
        let set_layout = device.create_descriptor_set_layout(&layout_info);
        DescriptorSetLayout::new(device, set_layout)
    }
}

pub struct PoolSizeRatio {
    pub descriptor_type: vk::DescriptorType,
    pub ratio: f32,
}
pub struct DescriptorAllocator {
    device: Arc<Device>,
    pool: Option<vk::DescriptorPool>,
}

impl DescriptorAllocator {
    pub fn new(device: Arc<Device>) -> DescriptorAllocator {
        Self { device, pool: None }
    }

    pub fn init_pool(&mut self, max_sets: u32, pool_ratios: &[PoolSizeRatio]) {
        let mut pool_sizes = Vec::with_capacity(pool_ratios.len());
        for pool_ratio in pool_ratios {
            pool_sizes.push(vk::DescriptorPoolSize {
                ty: pool_ratio.descriptor_type,
                descriptor_count: (max_sets as f32 * pool_ratio.ratio) as u32,
            });
        }
        let pool_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            max_sets,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            p_next: std::ptr::null(),
            ..Default::default()
        };
        self.pool = Some(self.device.create_descriptor_pool(&pool_info));
    }

    #[allow(dead_code)]
    pub fn clear_descriptors(&self) {
        if let Some(pool) = self.pool {
            self.device.reset_descriptor_pool(pool);
        } else {
            panic!("Tried to clear non-initialized descriptor pool");
        }
    }

    pub fn destroy_pool(&mut self) {
        if let Some(pool) = self.pool.take() {
            self.device.destroy_descriptor_pool(pool);
        } else {
            panic!("Tried to destroy non-initialized descriptor pool");
        }
    }

    //TODO: think of a solution to handle the dependency of the descriptor set to the pool (aka
    //make sure that descriptor set is invalidated when pool is reset/destroyed)
    pub fn allocate(&self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        if let Some(pool) = self.pool {
            let alloc_info = vk::DescriptorSetAllocateInfo {
                s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
                p_next: std::ptr::null(),
                descriptor_pool: pool,
                descriptor_set_count: 1,
                p_set_layouts: &layout,
                ..Default::default()
            };
            // we should only have one element in array since we only allocate one descriptor set
            self.device
                .allocate_descriptor_sets(&alloc_info)
                .expect("I pray that i never run out of memory")[0]
        } else {
            panic!("Tried to allocate from non-initialized descriptor pool");
        }
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        log::debug!("Destroying DescriptorAllocator");
        self.destroy_pool();
    }
}

pub struct DescriptorAllocatorGrowable {
    device: Arc<Device>,
    ratios: Vec<PoolSizeRatio>,
    full_pools: Vec<vk::DescriptorPool>,
    ready_pools: Vec<vk::DescriptorPool>,
    sets_per_pool: u32,
}

impl DescriptorAllocatorGrowable {
    pub fn new(device: Arc<Device>, ratios: Vec<PoolSizeRatio>, max_sets: u32) -> Self {
        Self {
            device,
            ratios,
            full_pools: Vec::new(),
            ready_pools: Vec::new(),
            sets_per_pool: max_sets,
        }
    }

    pub fn init_pool(&mut self) {
        let pool = self.create_new_pool(self.sets_per_pool, &self.ratios);
        self.ready_pools.push(pool);
        self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
    }

    pub fn clear_pools(&mut self) {
        self.ready_pools.append(&mut self.full_pools);
        for pool in self.ready_pools.iter() {
            self.device.reset_descriptor_pool(*pool);
        }
    }

    pub fn destroy_pools(&mut self) {
        for pool in self.ready_pools.iter() {
            self.device.destroy_descriptor_pool(*pool);
        }
        for pool in self.full_pools.iter() {
            self.device.destroy_descriptor_pool(*pool);
        }
        self.ready_pools.clear();
        self.full_pools.clear();
    }

    fn get_pool(&mut self) -> vk::DescriptorPool {
        if self.ready_pools.is_empty() {
            let new_pool = self.create_new_pool(self.sets_per_pool, &self.ratios);
            self.sets_per_pool = (self.sets_per_pool as f32 * 1.5) as u32;
            self.sets_per_pool = u32::min(self.sets_per_pool, 4092);
            new_pool
        } else {
            self.ready_pools
                .pop()
                .expect("Vector should not be empty since we just checked for it")
        }
    }

    fn create_new_pool(&self, set_count: u32, pool_ratios: &[PoolSizeRatio]) -> vk::DescriptorPool {
        let pool_sizes: Vec<vk::DescriptorPoolSize> = pool_ratios
            .iter()
            .map(|ratio| vk::DescriptorPoolSize {
                ty: ratio.descriptor_type,
                descriptor_count: (set_count as f32 * ratio.ratio) as u32,
            })
            .collect();

        let pool_create_info = vk::DescriptorPoolCreateInfo {
            s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: set_count,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        self.device.create_descriptor_pool(&pool_create_info)
    }

    pub fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> vk::DescriptorSet {
        let pool_to_use = self.get_pool();

        let mut alloc_info = vk::DescriptorSetAllocateInfo {
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: std::ptr::null(),
            descriptor_pool: pool_to_use,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            ..Default::default()
        };
        let result = self.device.allocate_descriptor_sets(&alloc_info);
        match result {
            Ok(sets) => {
                self.ready_pools.push(pool_to_use);
                sets[0]
            }
            Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) | Err(vk::Result::ERROR_FRAGMENTED_POOL) => {
                self.full_pools.push(pool_to_use);
                let pool_to_use = self.get_pool();
                alloc_info.descriptor_pool = pool_to_use;
                // just crash if it doesnt work the second time
                let sets = self
                    .device
                    .allocate_descriptor_sets(&alloc_info)
                    .expect("I pray that i never run out of memory")[0];
                self.ready_pools.push(pool_to_use);
                sets
            }
            _ => panic!("I pray that i never run out of memory"),
        }
    }
}

impl Drop for DescriptorAllocatorGrowable {
    fn drop(&mut self) {
        log::debug!("Destroying DescriptorAllocatorGrowable");
        self.destroy_pools();
    }
}

pub struct DescriptorWriter<'a> {
    //NOTE: box is used here to allow vector resizing without invalidating references
    // stored in self.writes
    #[allow(clippy::vec_box)]
    buffer_infos: Vec<Box<vk::DescriptorBufferInfo>>,
    #[allow(clippy::vec_box)]
    image_infos: Vec<Box<vk::DescriptorImageInfo>>,
    writes: Vec<vk::WriteDescriptorSet<'a>>,
}

impl<'a> DescriptorWriter<'a> {
    pub fn new() -> DescriptorWriter<'a> {
        DescriptorWriter {
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
        }
    }

    pub fn add_uniform_buffer(&mut self, binding: i32, buffer: vk::Buffer, size: u64, offset: u64) {
        self.add_buffer(
            binding,
            buffer,
            size,
            offset,
            vk::DescriptorType::UNIFORM_BUFFER,
        );
    }

    pub fn add_buffer(
        &mut self,
        binding: i32,
        buffer: vk::Buffer,
        size: u64,
        offset: u64,
        descriptor_type: vk::DescriptorType,
    ) {
        let buffer_info = vk::DescriptorBufferInfo {
            buffer,
            offset,
            range: size,
        };
        self.buffer_infos.push(Box::new(buffer_info));

        let descriptor_write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: std::ptr::null(),
            dst_set: vk::DescriptorSet::null(),
            dst_binding: binding as u32,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type,
            p_buffer_info: &**self
                .buffer_infos
                .last()
                .expect("Vector should have at least one element since we just added one"),
            ..Default::default()
        };
        self.writes.push(descriptor_write);
    }

    pub fn add_image(
        &mut self,
        binding: i32,
        image_view: vk::ImageView,
        sampler: vk::Sampler,
        image_layout: vk::ImageLayout,
        descriptor_type: vk::DescriptorType,
    ) {
        let image_info = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout,
        };
        self.image_infos.push(Box::new(image_info));

        let descriptor_write = vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: std::ptr::null(),
            dst_set: vk::DescriptorSet::null(),
            dst_binding: binding as u32,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type,
            p_image_info: &**self
                .image_infos
                .last()
                .expect("Vector should have at least one element since we just added one"),
            ..Default::default()
        };
        self.writes.push(descriptor_write);
    }

    pub fn add_storage_image(&mut self, binding: i32, image_view: vk::ImageView) {
        self.add_image(
            binding,
            image_view,
            vk::Sampler::null(),
            vk::ImageLayout::GENERAL,
            vk::DescriptorType::STORAGE_IMAGE,
        );
    }

    pub fn clear(&mut self) {
        self.buffer_infos.clear();
        self.image_infos.clear();
        self.writes.clear();
    }

    pub fn update_descriptor_set(&mut self, device: &Device, set: vk::DescriptorSet) {
        for write in self.writes.iter_mut() {
            write.dst_set = set;
        }
        device.update_descriptor_sets(&self.writes);
    }
}
