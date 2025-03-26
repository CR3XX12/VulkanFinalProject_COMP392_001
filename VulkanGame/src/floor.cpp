#include <iostream>
#include "floor.hpp"
#include "camera.hpp"  // to use Camera in updateUniformBuffer
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stdexcept>
#include <cstring>
#include <windows.h> // at the top for GetCurrentDirectoryA

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// Define floor geometry (a square plane on the X-Y plane)
static const std::vector<Vertex> FLOOR_VERTICES = {
    {{-25.0f, -25.0f, 0.0f}, {0.0f, 0.0f}},   // 0: bottom-left corner
    {{ 25.0f, -25.0f, 0.0f}, {10.0f, 0.0f}},  // 1: bottom-right (UV x=10 for tiling)
    {{ 25.0f,  25.0f, 0.0f}, {10.0f, 10.0f}}, // 2: top-right
    {{-25.0f,  25.0f, 0.0f}, {0.0f, 10.0f}}   // 3: top-left
};
static const std::vector<uint16_t> FLOOR_INDICES = {
    0, 1, 2,  2, 3, 0  // two triangles (0-1-2 and 2-3-0)
};

// Initialize static texture path
const std::string Floor::TEXTURE_PATH = "textures/grass.png";

Floor::Floor(vk::Device device, vk::PhysicalDevice physicalDevice, vk::CommandPool commandPool, vk::Queue graphicsQueue, uint32_t swapChainImageCount)
    : device(device), physicalDevice(physicalDevice), commandPool(commandPool), graphicsQueue(graphicsQueue) {
    createVertexBuffer();
    createIndexBuffer();
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    createDescriptorSetLayout();
    createUniformBuffers(swapChainImageCount);
    createDescriptorPool(swapChainImageCount);
    createDescriptorSets(swapChainImageCount);
}

Floor::~Floor() {
    // Clean up allocated resources
    for (size_t i = 0; i < uniformBuffers.size(); ++i) {
        device.unmapMemory(uniformBuffersMemory[i]);
        device.freeMemory(uniformBuffersMemory[i]);
        device.destroyBuffer(uniformBuffers[i]);
    }
    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroySampler(textureSampler);
    device.destroyImageView(textureImageView);
    device.destroyImage(textureImage);
    device.freeMemory(textureImageMemory);
    device.destroyBuffer(indexBuffer);
    device.freeMemory(indexBufferMemory);
    device.destroyBuffer(vertexBuffer);
    device.freeMemory(vertexBufferMemory);
}

uint32_t Floor::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void Floor::createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(FLOOR_VERTICES[0]) * FLOOR_VERTICES.size();

    // Create a staging buffer (CPU-visible) to load vertex data
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stagingBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
    // Copy vertex data into the staging buffer
    void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
    std::memcpy(data, FLOOR_VERTICES.data(), (size_t) bufferSize);
    device.unmapMemory(stagingBufferMemory);

    // Create the device-local vertex buffer 
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
    vertexBuffer = device.createBuffer(bufferInfo);
    memRequirements = device.getBufferMemoryRequirements(vertexBuffer);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    vertexBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(vertexBuffer, vertexBufferMemory, 0);

    // Copy the data from staging buffer to the vertex buffer using a command buffer
    vk::CommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.level = vk::CommandBufferLevel::ePrimary;
    allocInfoCmd.commandPool = commandPool;
    allocInfoCmd.commandBufferCount = 1;
    vk::CommandBuffer copyCmd = device.allocateCommandBuffers(allocInfoCmd)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    copyCmd.begin(beginInfo);
    vk::BufferCopy copyRegion{0, 0, bufferSize};
    copyCmd.copyBuffer(stagingBuffer, vertexBuffer, copyRegion);
    copyCmd.end();
    // Submit to graphics queue and wait
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCmd;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
    // Free temporary resources
    device.freeCommandBuffers(commandPool, copyCmd);
    device.freeMemory(stagingBufferMemory);
    device.destroyBuffer(stagingBuffer);
}

void Floor::createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(FLOOR_INDICES[0]) * FLOOR_INDICES.size();
    // Create staging buffer for indices
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stagingBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
    // Copy index data
    void* data = device.mapMemory(stagingBufferMemory, 0, bufferSize);
    std::memcpy(data, FLOOR_INDICES.data(), (size_t) bufferSize);
    device.unmapMemory(stagingBufferMemory);

    // Create device-local index buffer
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
    indexBuffer = device.createBuffer(bufferInfo);
    memRequirements = device.getBufferMemoryRequirements(indexBuffer);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(indexBuffer, indexBufferMemory, 0);

    // Copy indices from staging to index buffer
    vk::CommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.level = vk::CommandBufferLevel::ePrimary;
    allocInfoCmd.commandPool = commandPool;
    allocInfoCmd.commandBufferCount = 1;
    vk::CommandBuffer copyCmd = device.allocateCommandBuffers(allocInfoCmd)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    copyCmd.begin(beginInfo);
    vk::BufferCopy copyRegion{0, 0, bufferSize};
    copyCmd.copyBuffer(stagingBuffer, indexBuffer, copyRegion);
    copyCmd.end();
    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCmd;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();
    device.freeCommandBuffers(commandPool, copyCmd);
    device.freeMemory(stagingBufferMemory);
    device.destroyBuffer(stagingBuffer);
}

void Floor::createTextureImage() {
    // Load the image pixels using stb_image (RGBA8)
    int texWidth, texHeight, texChannels;
    char buffer[MAX_PATH];
GetCurrentDirectoryA(MAX_PATH, buffer);
std::cout << "Working directory: " << buffer << std::endl;
std::cout << "Trying to load texture: " << TEXTURE_PATH << std::endl;

    stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("failed to load texture image!");
    }
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    // Create a host-visible staging buffer for the pixel data
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = imageSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::Buffer stagingBuffer = device.createBuffer(bufferInfo);
    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(stagingBuffer);
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, 
                                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vk::DeviceMemory stagingBufferMemory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(stagingBuffer, stagingBufferMemory, 0);
    // Copy image pixels into the staging buffer
    void* data = device.mapMemory(stagingBufferMemory, 0, imageSize);
    std::memcpy(data, pixels, static_cast<size_t>(imageSize));
    device.unmapMemory(stagingBufferMemory);
    stbi_image_free(pixels);  // free original pixel data

    // Create the Vulkan image (device local) for the texture
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.extent.width = static_cast<uint32_t>(texWidth);
    imageInfo.extent.height = static_cast<uint32_t>(texHeight);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = vk::Format::eR8G8B8A8Srgb;  // using SRGB format for texture
    imageInfo.tiling = vk::ImageTiling::eOptimal;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;
    imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    textureImage = device.createImage(imageInfo);
    memRequirements = device.getImageMemoryRequirements(textureImage);
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    textureImageMemory = device.allocateMemory(allocInfo);
    device.bindImageMemory(textureImage, textureImageMemory, 0);

    // Copy pixel data from staging buffer to texture image 
    // (including transitioning image layout)
    vk::CommandBufferAllocateInfo allocInfoCmd{};
    allocInfoCmd.level = vk::CommandBufferLevel::ePrimary;
    allocInfoCmd.commandPool = commandPool;
    allocInfoCmd.commandBufferCount = 1;
    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfoCmd)[0];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);
    // Transition image layout from undefined to transfer-destination optimal
    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = vk::ImageLayout::eUndefined;
    barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = textureImage;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = {};  // nothing to wait on
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlags(), nullptr, nullptr, barrier
    );
    // Copy buffer to image
    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    vk::Offset3D offset;
    offset.x = 0;
    offset.y = 0;
    offset.z = 0;
    region.imageOffset = offset;
    vk::Extent3D extent;
    extent.width = static_cast<uint32_t>(texWidth);
    extent.height = static_cast<uint32_t>(texHeight);
    extent.depth = 1;
    region.imageExtent = extent;

    commandBuffer.copyBufferToImage(stagingBuffer, textureImage, vk::ImageLayout::eTransferDstOptimal, region);
    // Transition image layout from transfer-dst to shader-read (for sampling in shader)
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
    commandBuffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        vk::DependencyFlags(), nullptr, nullptr, barrier
    );
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    graphicsQueue.submit(submitInfo);
    graphicsQueue.waitIdle();

    // Free staging buffer resources
    device.freeCommandBuffers(commandPool, commandBuffer);
    device.freeMemory(stagingBufferMemory);
    device.destroyBuffer(stagingBuffer);
}

void Floor::createTextureImageView() {
    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = textureImage;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.format = vk::Format::eR8G8B8A8Srgb;
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    textureImageView = device.createImageView(viewInfo);
}

void Floor::createTextureSampler() {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.anisotropyEnable = physicalDevice.getFeatures().samplerAnisotropy;
    samplerInfo.maxAnisotropy = samplerInfo.anisotropyEnable ? properties.limits.maxSamplerAnisotropy : 1.0f;
    samplerInfo.borderColor = vk::BorderColor::eIntOpaqueBlack;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;
    textureSampler = device.createSampler(samplerInfo);
}

void Floor::createDescriptorSetLayout() {
    // Binding 0: Uniform buffer (MVP matrix), Binding 1: Combined image sampler (texture)
    vk::DescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.stageFlags = vk::ShaderStageFlagBits::eFragment;
    samplerLayoutBinding.pImmutableSamplers = nullptr;

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = { uboLayoutBinding, samplerLayoutBinding };
    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

void Floor::createUniformBuffers(uint32_t swapChainImageCount) {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    uniformBuffers.resize(swapChainImageCount);
    uniformBuffersMemory.resize(swapChainImageCount);
    uniformBuffersMapped.resize(swapChainImageCount);

    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = bufferSize;
    bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    for (uint32_t i = 0; i < swapChainImageCount; i++) {
        uniformBuffers[i] = device.createBuffer(bufferInfo);
        vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(uniformBuffers[i]);
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memReq.size;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, 
                                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        uniformBuffersMemory[i] = device.allocateMemory(allocInfo);
        device.bindBufferMemory(uniformBuffers[i], uniformBuffersMemory[i], 0);
        // Map memory and keep it mapped (coherent memory) for easy updates
        uniformBuffersMapped[i] = device.mapMemory(uniformBuffersMemory[i], 0, bufferSize);
    }
}

void Floor::createDescriptorPool(uint32_t swapChainImageCount) {
    std::array<vk::DescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = vk::DescriptorType::eUniformBuffer;
    poolSizes[0].descriptorCount = swapChainImageCount;
    poolSizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSizes[1].descriptorCount = swapChainImageCount;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.maxSets = swapChainImageCount;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    descriptorPool = device.createDescriptorPool(poolInfo);
}

void Floor::createDescriptorSets(uint32_t swapChainImageCount) {
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImageCount, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = swapChainImageCount;
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets = device.allocateDescriptorSets(allocInfo);

    for (uint32_t i = 0; i < swapChainImageCount; i++) {
        vk::DescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(UniformBufferObject);

        vk::DescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        imageInfo.imageView = textureImageView;
        imageInfo.sampler = textureSampler;

        std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &bufferInfo;

        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        device.updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void Floor::bind(vk::CommandBuffer commandBuffer) {
    vk::DeviceSize offsets[] = {0};
    commandBuffer.bindVertexBuffers(0, 1, &vertexBuffer, offsets);
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
}

void Floor::draw(vk::CommandBuffer commandBuffer) {
    commandBuffer.drawIndexed(static_cast<uint32_t>(FLOOR_INDICES.size()), 1, 0, 0, 0);
}

void Floor::updateUniformBuffer(uint32_t currentImage, const Camera& camera) {
    // Prepare the transformation matrices for this frame
    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);  // floor is static at origin
    ubo.view = camera.getViewMatrix();
    ubo.proj = camera.getProjectionMatrix();
    // Copy data to mapped uniform buffer (coherent memory so no flush needed)
    std::memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}
