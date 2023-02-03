#include "Shader.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>
#include <glm/glm.hpp>

Shader::Shader(const std::string& vertexPath, const std::string& fragmentPath)
    : m_vShaderPath(vertexPath), m_fShaderPath(fragmentPath), m_RendererID(0)
{
    // 1. retrieve the vertex/fragment source code from filePath
    std::string vertexCode;
    std::string fragmentCode;
    std::ifstream vShaderFile;
    std::ifstream fShaderFile;
    // ensure ifstream objects can throw exceptions:
    vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    try
    {
        // open files
        vShaderFile.open(vertexPath);
        fShaderFile.open(fragmentPath);
        std::stringstream vShaderStream, fShaderStream;
        // read file's buffer contents into streams
        vShaderStream << vShaderFile.rdbuf();
        fShaderStream << fShaderFile.rdbuf();
        // close file handlers
        vShaderFile.close();
        fShaderFile.close();
        // convert stream into string
        vertexCode = vShaderStream.str();
        fragmentCode = fShaderStream.str();
    }
    catch (std::ifstream::failure& e)
    {
        std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
    }

    m_RendererID = CreateShader(vertexCode, fragmentCode);
    /*
    const std::string vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();

    // 2. compile shaders
    unsigned int vertex, fragment;
    // vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");
    std::cout << "Vertex Shader compiled" << std::endl << vShaderCode << std::endl;
    // fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");
    std::cout << "Fragment Shader compiled" << std::endl << fShaderCode << std::endl;

    // shader Program
    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");
    // delete the shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertex);
    glDeleteShader(fragment);*/
}

Shader::~Shader() 
{
    glDeleteProgram(m_RendererID);
}

void Shader::Bind() const
{
    glUseProgram(m_RendererID);
}

void Shader::Unbind() const
{
    glUseProgram(0);
}

void Shader::SetUniform4f(const std::string& name, float v0, float v1, float v2, float v3)
{
    glUniform4f(GetUniformLocation(name), v0, v1, v2, v3);
}

void Shader::SetUniform1i(const std::string& name, int value)
{
    glUniform1i(GetUniformLocation(name), value);
}


void Shader::SetUniformMat4f(const std::string& name,const glm::mat4& matrix)
{
    glUniformMatrix4fv(GetUniformLocation(name), 1, GL_FALSE, &matrix[0][0]);
}


unsigned int Shader::CreateShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompileShader(GL_FRAGMENT_SHADER, fragmentShader);
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    /*ERROR HANDLING*/
    int result;
    glGetProgramiv(program, GL_LINK_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(sizeof(char) * length);
        glGetProgramInfoLog(program, length, &length, message);
        std::cout << "ERROR! The shader program isn't linked correctly." << std::endl;
        std::cout << "[OpenGL Error Message]: " << message << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    else
    {
    //    std::cout << "Vertex Shader: " << vertexShader << std::endl;
    //    std::cout << "Fragment Shader: " << fragmentShader << std::endl;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

unsigned int Shader::CompileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    /* Error Handling */
    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "ERROR! The " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
            << " shader hasn't been compiled successfully." << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }
    else
        std::cout << "The " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
        << " shader compiled." << std::endl;
    return id;
}

int Shader::GetUniformLocation(const std::string& name)
{
    /* Check if Cache contains*/
    if (m_UniformLocationCache.find(name) != m_UniformLocationCache.end())
        return m_UniformLocationCache[name];

    int location = glGetUniformLocation(m_RendererID, name.c_str());
    if (location == -1)
        std::cout << "warning: uniform '" << name << "' doesn''t exist!" << std::endl;
    
    /* Add to Cache */
    m_UniformLocationCache[name] = location;
    
    return location;
}