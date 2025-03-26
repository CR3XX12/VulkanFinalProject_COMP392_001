#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <memory>
#include "floor.hpp"
#include "sky.hpp"
#include "camera.hpp"

class VulkanApp {
public:
    VulkanApp();
    ~VulkanApp();
    void run();

private:
    // Window and Vulkan base objects
    GLFWwindow* window;
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::RenderPass renderPass;
    vk::PipelineLayout floorPipelineLayout;
    vk::PipelineLayout skyPipelineLayout;
    vk::Pipeline floorPipeline;
    vk::Pipeline skyPipeline;
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;
    static const int MAX_FRAMES_IN_FLIGHT = 2;

    // Game scene components
    std::unique_ptr<Floor> floor;
    std::unique_ptr<Sky> sky;
    Camera camera;

    // Initialization and setup methods
    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createCommandPool();
    void createGraphicsPipelines();
    void createFramebuffers();
    void createCommandBuffers();
    void createSyncObjects();

    // Main loop methods
    void drawFrame();
};
