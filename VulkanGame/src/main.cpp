#include "vulkan_app.hpp"
#include <iostream>
#include <cstdlib>
#include <exception>

int main() {
    try {
        VulkanApp app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
