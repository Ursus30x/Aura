#include <glad/gl.h> 
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "objLoader.h"

// --- Multi-Target Morph Vertex Shader (Supports 5 Targets) ---
const char* vertexShaderSource = R"(
#version 450 core

// 1. Base Model (Neutral) - Loc 0, 1, 2
layout (location = 0) in vec3 aPosBase;
layout (location = 1) in vec3 aNormalBase;
layout (location = 2) in vec2 aTexCoord; 

// 2. Mouth Open - Loc 3, 4
layout (location = 3) in vec3 aPosMouth;
layout (location = 4) in vec3 aNormalMouth;

// 3. Left Eye Full - Loc 5, 6
layout (location = 5) in vec3 aPosLEyeFull;
layout (location = 6) in vec3 aNormalLEyeFull;

// 4. Left Eye Half - Loc 7, 8
layout (location = 7) in vec3 aPosLEyeHalf;
layout (location = 8) in vec3 aNormalLEyeHalf;

// 5. Right Eye Full - Loc 9, 10
layout (location = 9) in vec3 aPosREyeFull;
layout (location = 10) in vec3 aNormalREyeFull;

// 6. Right Eye Half - Loc 11, 12
layout (location = 11) in vec3 aPosREyeHalf;
layout (location = 12) in vec3 aNormalREyeHalf;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

// Weights
uniform float wMouth;
uniform float wLEyeFull;
uniform float wLEyeHalf;
uniform float wREyeFull;
uniform float wREyeHalf;

void main() {
    // Calculate Deltas (Target - Base)
    vec3 dMouth    = aPosMouth    - aPosBase;
    vec3 dLEyeFull = aPosLEyeFull - aPosBase;
    vec3 dLEyeHalf = aPosLEyeHalf - aPosBase;
    vec3 dREyeFull = aPosREyeFull - aPosBase;
    vec3 dREyeHalf = aPosREyeHalf - aPosBase;

    // Sum Positions
    vec3 finalPos = aPosBase 
                  + (dMouth * wMouth)
                  + (dLEyeFull * wLEyeFull) + (dLEyeHalf * wLEyeHalf)
                  + (dREyeFull * wREyeFull) + (dREyeHalf * wREyeHalf);

    // Calculate Normal Deltas
    vec3 nMouth    = aNormalMouth    - aNormalBase;
    vec3 nLEyeFull = aNormalLEyeFull - aNormalBase;
    vec3 nLEyeHalf = aNormalLEyeHalf - aNormalBase;
    vec3 nREyeFull = aNormalREyeFull - aNormalBase;
    vec3 nREyeHalf = aNormalREyeHalf - aNormalBase;

    // Sum Normals
    vec3 finalNorm = normalize(aNormalBase 
                  + (nMouth * wMouth)
                  + (nLEyeFull * wLEyeFull) + (nLEyeHalf * wLEyeHalf)
                  + (nREyeFull * wREyeFull) + (nREyeHalf * wREyeHalf));

    FragPos = vec3(model * vec4(finalPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * finalNorm;  
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)";

const char* fragmentShaderSource = R"(
#version 450 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 objectColor;
uniform vec3 lightColor;
uniform vec3 lightPos;

void main() {
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;
  
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    vec3 result = (ambient + diffuse) * objectColor;
    FragColor = vec4(result, 1.0);
}
)";

GLuint CreateShader(const char* vSource, const char* fSource) {
    auto compile = [](GLenum type, const char* src) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &src, NULL);
        glCompileShader(shader);
        return shader;
    };
    GLuint vs = compile(GL_VERTEX_SHADER, vSource);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fSource);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Helper to load into vector and print status
bool loadMesh(const char* path, std::vector<Vertex>& mesh) {
    std::cout << "Loading: " << path << "... ";
    if (loadOBJ(path, mesh)) {
        std::cout << "OK (" << mesh.size() << " v)" << std::endl;
        return true;
    }
    return false;
}

// Helper to setup VBO for a target (Pos + Normal only)
void setupTargetVBO(GLuint& VBO, const std::vector<Vertex>& mesh, int locPos, int locNorm) {
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, mesh.size() * sizeof(Vertex), mesh.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(locPos, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
    glEnableVertexAttribArray(locPos);
    
    glVertexAttribPointer(locNorm, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
    glEnableVertexAttribArray(locNorm);
}

int main(int argc, char** argv) {
    // We expect 1 Base + 5 Targets = 6 files + 1 executable arg = 7 argc
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <base> <mouth> <L_Full> <L_Half> <R_Full> <R_Half>" << std::endl;
        return -1;
    }

    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720, "Eye Blink Correction", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (gladLoadGL(glfwGetProcAddress) == 0) return -1;
    glViewport(0, 0, 1280, 720);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    // --- LOAD MODELS ---
    std::vector<Vertex> base, mouth, lFull, lHalf, rFull, rHalf;
    
    if (!loadMesh(argv[1], base)) return -1;
    if (!loadMesh(argv[2], mouth)) return -1;
    if (!loadMesh(argv[3], lFull)) return -1;
    if (!loadMesh(argv[4], lHalf)) return -1;
    if (!loadMesh(argv[5], rFull)) return -1;
    if (!loadMesh(argv[6], rHalf)) return -1;

    // Topology Check
    size_t count = base.size();
    if (mouth.size() != count || lFull.size() != count || lHalf.size() != count || rFull.size() != count || rHalf.size() != count) {
        std::cerr << "Error: Vertex counts mismatch!" << std::endl;
        return -1;
    }

    // --- GPU BUFFERS ---
    GLuint VAO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // 1. Base (0, 1, 2) - Includes UVs
    GLuint VBO_Base;
    glGenBuffers(1, &VBO_Base);
    glBindBuffer(GL_ARRAY_BUFFER, VBO_Base);
    glBufferData(GL_ARRAY_BUFFER, base.size() * sizeof(Vertex), base.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0); // Pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal)); // Norm
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, texUV)); // UV
    glEnableVertexAttribArray(2);

    // 2. Targets (Pos + Norm only)
    GLuint VBO_Mouth, VBO_LF, VBO_LH, VBO_RF, VBO_RH;
    
    setupTargetVBO(VBO_Mouth, mouth, 3, 4);
    setupTargetVBO(VBO_LF,    lFull, 5, 6);
    setupTargetVBO(VBO_LH,    lHalf, 7, 8);
    setupTargetVBO(VBO_RF,    rFull, 9, 10);
    setupTargetVBO(VBO_RH,    rHalf, 11, 12);

    GLuint shaderProgram = CreateShader(vertexShaderSource, fragmentShaderSource);
    
    // App State
    glEnable(GL_DEPTH_TEST);
    glm::vec3 clear_color(0.2f);
    glm::vec3 mesh_color(1.0f, 0.5f, 0.2f);
    glm::vec3 camera_view(0.0f, 0.0f, -4.0f);
    
    bool autoRotate = false;
    float rotationSpeed = 1.0f;
    float manualRotation = 0.0f;

    // --- User Sliders (0.0 to 1.0) ---
    float userMouth = 0.0f;
    float userLeftEye = 0.0f;
    float userRightEye = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Face Controller"); 
            ImGui::Text("Intermediate Shape Logic Active");
            ImGui::Separator();
            
            // 1. Mouth Slider
            ImGui::SliderFloat("Mouth Open", &userMouth, 0.0f, 1.0f);

            // 2. Eye Sliders (These drive the intermediate logic)
            
            ImGui::SliderFloat("Left Eye Blink", &userLeftEye, 0.0f, 1.0f);
            ImGui::SliderFloat("Right Eye Blink", &userRightEye, 0.0f, 1.0f);
            
            ImGui::Separator();
            ImGui::Text("Scene Settings");
            ImGui::ColorEdit3("Mesh Color", (float*)&mesh_color);
            ImGui::DragFloat3("Camera", (float*)&camera_view, 0.1f);
            
            ImGui::Checkbox("Auto Rotate", &autoRotate);
            if (autoRotate) ImGui::SliderFloat("Speed", &rotationSpeed, 0.0f, 5.0f);
            else ImGui::SliderFloat("Rotation", &manualRotation, 0.0f, 360.0f);
            
            ImGui::End();
        }

        ImGui::Render();
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(clear_color.x, clear_color.y, clear_color.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // --- INTERMEDIATE SHAPE LOGIC ---
        auto calculateEyeWeights = [](float slider, float& wHalf, float& wFull) {
            if (slider <= 0.5f) {
                // Phase 1: Base -> Half
                // Slider 0.0 -> wHalf 0.0
                // Slider 0.5 -> wHalf 1.0
                wHalf = slider * 2.0f;
                wFull = 0.0f;
            } else {
                // Phase 2: Half -> Full
                // Slider 0.5 -> wHalf 1.0, wFull 0.0
                // Slider 1.0 -> wHalf 0.0, wFull 1.0
                float t = (slider - 0.5f) * 2.0f;
                wHalf = 1.0f - t;
                wFull = t;
            }
        };

        float wLF, wLFull, wRF, wRFull;
        calculateEyeWeights(userLeftEye, wLF, wLFull);
        calculateEyeWeights(userRightEye, wRF, wRFull);
        // --------------------------------

        // Uniforms
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)w / (float)h, 0.1f, 100.0f);
        glm::mat4 view = glm::translate(glm::mat4(1.0f), camera_view);
        glm::mat4 model = glm::mat4(1.0f);

        float finalAngle = autoRotate ? (float)glfwGetTime() * rotationSpeed : glm::radians(manualRotation);
        model = glm::rotate(model, finalAngle, glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view[0][0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model[0][0]);
        
        // Pass Computed Weights
        glUniform1f(glGetUniformLocation(shaderProgram, "wMouth"),    userMouth);
        glUniform1f(glGetUniformLocation(shaderProgram, "wLEyeFull"), wLFull);
        glUniform1f(glGetUniformLocation(shaderProgram, "wLEyeHalf"), wLF);
        glUniform1f(glGetUniformLocation(shaderProgram, "wREyeFull"), wRFull);
        glUniform1f(glGetUniformLocation(shaderProgram, "wREyeHalf"), wRF);
        
        glUniform3fv(glGetUniformLocation(shaderProgram, "objectColor"), 1, &mesh_color[0]);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);
        glUniform3f(glGetUniformLocation(shaderProgram, "lightPos"), 2.0f, 2.0f, 2.0f);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, base.size());

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO_Base);
    glDeleteBuffers(1, &VBO_Mouth);
    glDeleteBuffers(1, &VBO_LF);
    glDeleteBuffers(1, &VBO_LH);
    glDeleteBuffers(1, &VBO_RF);
    glDeleteBuffers(1, &VBO_RH);
    glDeleteProgram(shaderProgram);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}