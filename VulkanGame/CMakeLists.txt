cmake_minimum_required(VERSION 3.10)
project(VulkanGame)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Vulkan (requires Vulkan SDK)
find_package(Vulkan REQUIRED)

# Find packages installed via vcpkg
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/vulkan_app.cpp
    src/floor.cpp
    src/sky.cpp
)

# Header files (not required but helpful for IDEs)
set(HEADERS
    include/vulkan_app.hpp
    include/floor.hpp
    include/sky.hpp
    include/camera.hpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE 
    Vulkan::Vulkan 
    glfw 
    glm::glm
)

# Copy shaders and textures into the build output directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR}/Release)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/textures DESTINATION ${CMAKE_BINARY_DIR}/Release)


