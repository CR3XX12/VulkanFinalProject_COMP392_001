#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <array>
#include <vector>
#include <string>

// Vertex structure for floor vertices (position and UV)
struct Vertex {
    glm::vec3 pos;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

class Floor {
public:
    Floor(vk::Device device, vk::PhysicalDevice physicalDevice, vk::CommandPool commandPool, vk::Queue graphicsQueue, uint32_t swapChainImageCount);
    ~Floor();

    vk::DescriptorSetLayout getDescriptorSetLayout() const { return descriptorSetLayout; }
    std::vector<vk::DescriptorSet> getDescriptorSets() const { return descriptorSets; }

    void bind(vk::CommandBuffer commandBuffer);
    void draw(vk::CommandBuffer commandBuffer);
    void updateUniformBuffer(uint32_t currentImage, const struct Camera& camera);

    // Path to the floor texture image
    static const std::string TEXTURE_PATH;

private:
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::CommandPool commandPool;
    vk::Queue graphicsQueue;

    // Buffers for floor geometry
    vk::Buffer vertexBuffer;
    vk::DeviceMemory vertexBufferMemory;
    vk::Buffer indexBuffer;
    vk::DeviceMemory indexBufferMemory;

    // Texture image and sampler
    vk::Image textureImage;
    vk::DeviceMemory textureImageMemory;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    // Descriptor set layout and sets for the uniform+texture
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    // Uniform buffers for each swap-chain image (MVP matrix)
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    // Helper functions for resource creation
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void createVertexBuffer();
    void createIndexBuffer();
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void createDescriptorSetLayout();
    void createUniformBuffers(uint32_t swapChainImageCount);
    void createDescriptorPool(uint32_t swapChainImageCount);
    void createDescriptorSets(uint32_t swapChainImageCount);
};
