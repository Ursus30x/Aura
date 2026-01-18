cmake_minimum_required(VERSION 3.10)
project(AuraAnimationModule)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

set(GLAD_DIR "vendor/glad")
set(GLAD_SOURCE "${GLAD_DIR}/src/gl.c")

set(IMGUI_DIR "vendor/imgui")
file(GLOB IMGUI_SOURCES 
    "${IMGUI_DIR}/*.cpp" 
    "${IMGUI_DIR}/*.h"
)
list(APPEND IMGUI_SOURCES 
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
)

add_executable(App src/main.cpp ${IMGUI_SOURCES} ${GLAD_SOURCE})

target_include_directories(App PRIVATE 
    ${IMGUI_DIR}
    "${GLAD_DIR}/include"
)

target_link_libraries(App 
    glfw
    GL 
    dl
)