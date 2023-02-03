#pragma once

#include <string>
#include <unordered_map>
#include <glm/mat4x4.hpp>

class Shader
{
private:
    std::string m_vShaderPath;
    std::string m_fShaderPath;

    unsigned int m_RendererID;

    std::unordered_map<std::string, int> m_UniformLocationCache;
    // caching for uniforms
public:
    Shader(const std::string& vertexPath, const std::string& fragmentPath);
    ~Shader();

    void Bind() const;   //gluseProgram
    void Unbind() const;

    /* set uniforms */
    void SetUniform4f(const std::string& name, float v0, float v1, float f2, float f3);
    void SetUniform1i(const std::string& name, int value);
    void SetUniformMat4f(const std::string& name, const glm::mat4& matrix);

private:
    unsigned int CompileShader(unsigned int type, const std::string& source);
    unsigned int CreateShader(const std::string& vertexShader, const std::string& fragmentShader);

    int GetUniformLocation(const std::string& name);
};
