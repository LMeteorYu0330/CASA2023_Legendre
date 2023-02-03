#define GLFW_INCLUDE_NONE


#include "config.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stdlib.h>
#include <stdio.h>


#include <IL/ilu.h>
#include <IL/ilut.h>

#include "camera.h"
#include "shader.h"
#include "sim.h"

#pragma comment(lib, "DevIL.lib")
#pragma comment(lib, "ILU.lib")
#pragma comment(lib, "ILUT.lib")

constexpr int width = 256 * 4;
constexpr int height = 256 * 4;

Camera camera(glm::vec3(0.0f, 0.1f, 0.55f));
float lastX = width / 2.0f;
float lastY = height / 2.0f;
bool firstMouse = true;

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, 0.012f);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, 0.012f);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, 0.012f);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, 0.012f);
	if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
		camera.ProcessKeyboard(UP, 0.012f);
	if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
		camera.ProcessKeyboard(DOWN, 0.012f);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = (float)xpos;
		lastY = (float)ypos;
		firstMouse = false;
	}


	float xoffset = (float)xpos - lastX;
	float yoffset = lastY - (float)ypos;

	lastX = (float)xpos;
	lastY = (float)ypos;

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
	{
		camera.ProcessMouseMovement(-xoffset, -yoffset);
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
}


int main(void)
{
	HWND consoleWindow = GetConsoleWindow();
	SetWindowPos(consoleWindow, 0, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOZORDER);


	init();

	GLFWwindow* window;


	if (!glfwInit())
		exit(EXIT_FAILURE);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	window = glfwCreateWindow(width * 2, height, "Simple example", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}

	glfwSetErrorCallback(error_callback);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetKeyCallback(window, key_callback);

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// NOTE: OpenGL error checks have been omitted for brevity
	
	glfwSwapInterval(0);

	gladLoadGL();

	ilInit();
	iluInit();
	ilEnable(IL_FILE_OVERWRITE);
	ilutRenderer(ILUT_OPENGL);

	ILuint imageID;
	ilGenImages(1, &imageID);



	Shader shader("shaders/vertex.shader", "shaders/fragment.shader");
	shader.Bind();

	//printf("0: %s\n", (char*)gluErrorString(glGetError()));

	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_ALPHA_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	std::vector<glm::vec3> vertices_l(res*res*res, glm::vec3(0));
	std::vector<glm::vec3> vertices_r(res*res*res, glm::vec3(0));
	int cnt = 0;
	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++)
			for (int k = 0; k < res; k++)
			{
				vertices_l[cnt] = glm::vec3(0.2f * i / res - 0.05f - 0.2f, 0.2f * j / res, 0.2f * k / res);
				vertices_r[cnt] = glm::vec3(0.2f * i / res + 0.05f + 0.0f, 0.2f * j / res, 0.2f * k / res);
				cnt++;
			}
	std::vector<float> colors_l[3];
	std::vector<float> colors_r[3];
	
	GLuint vao_l, vao_r;
	GLuint vbo_l, vbo_r, cbo_l[3], cbo_r[3];

	glGenVertexArrays(1, &vao_l);
	glBindVertexArray(vao_l);
	{
		glGenBuffers(1, &vbo_l);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_l);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices_l.size(), vertices_l.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		for (int i = 0; i < 3; i++)
		{
			colors_l[i].resize(res*res*res, 1.0f);
			glGenBuffers(1, &cbo_l[i]);
			glBindBuffer(GL_ARRAY_BUFFER, cbo_l[i]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * colors_l[i].size(), colors_l[i].data(), GL_DYNAMIC_DRAW);
			glVertexAttribPointer(1 + i, 1, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(1 + i);
		}
	}
	glBindVertexArray(0);

	glGenVertexArrays(1, &vao_r);
	glBindVertexArray(vao_r);
	{
		glGenBuffers(1, &vbo_r);
		glBindBuffer(GL_ARRAY_BUFFER, vbo_r);
		glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertices_r.size(), vertices_r.data(), GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		for (int i = 0; i < 3; i++)
		{
			colors_r[i].resize(res*res*res, 1.0f);
			glGenBuffers(1, &cbo_r[i]);
			glBindBuffer(GL_ARRAY_BUFFER, cbo_r[i]);
			glBufferData(GL_ARRAY_BUFFER, sizeof(float) * colors_r[i].size(), colors_r[i].data(), GL_DYNAMIC_DRAW);
			glVertexAttribPointer(1 + i, 1, GL_FLOAT, GL_FALSE, 0, 0);
			glEnableVertexAttribArray(1 + i);
		}
	}
	glBindVertexArray(0);

	//printf("3: %s\n", (char*)gluErrorString(glGetError()));

	glClearColor(0.2f, 0.2f, 0.2f, 0);
	while (!glfwWindowShouldClose(window))
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// pass projection matrix to shader (note that in this case it could change every frame)
		glm::mat4 projectionMatrix = glm::perspective(glm::radians(camera.Zoom), 2.0f * width / height, 0.1f, 100.0f);
		shader.SetUniformMat4f("projMat", projectionMatrix);

		// camera/view transformation
		glm::mat4 viewMatrix = camera.GetViewMatrix();
		shader.SetUniformMat4f("viewMat", viewMatrix);

		// for each object
		glm::mat4 worldMatrix = glm::mat4(1.0f); // for now
		shader.SetUniformMat4f("worldMat", worldMatrix);

		auto& velocityX = getVelocityX();
		auto& velocityY = getVelocityY();
		auto& velocityZ = getVelocityZ();
		auto& nodeVelocityX = getNodeVelocityX();
		auto& nodeVelocityY = getNodeVelocityY();
		auto& nodeVelocityZ = getNodeVelocityZ();

		int cnt = 0;

		glBindBuffer(GL_ARRAY_BUFFER, cbo_l[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * velocityX.size(), velocityX.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, cbo_l[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * velocityY.size(), velocityY.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, cbo_l[2]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * velocityZ.size(), velocityZ.data(), GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, cbo_r[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nodeVelocityX.size(), nodeVelocityX.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, cbo_r[1]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nodeVelocityY.size(), nodeVelocityY.data(), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, cbo_r[2]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * nodeVelocityZ.size(), nodeVelocityZ.data(), GL_DYNAMIC_DRAW);


		glBindVertexArray(vao_l);
		glDrawArrays(GL_POINTS, 0, vertices_l.size());


		glBindVertexArray(vao_r);
		glDrawArrays(GL_POINTS, 0, vertices_r.size());


	//	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
	//	glDrawPixels(velocities.size() * 2, velocities.size(), GL_RGB, GL_FLOAT, buffer);



		if (fit() == true)
		{
	//		step();


		//	ilBindImage(imageID);
		//	if (ilutGLScreen() == false)
		//	{
		//		ilDeleteImages(1, &imageID);
		//		return false;
		//	}
		//	if (ilEnable(IL_FILE_OVERWRITE) == false)
		//	{
		//		ilDeleteImages(1, &imageID);
		//		return false;
		//	}
		//	char bufferString[256];
		//	static int num = 0;
		//	sprintf_s(bufferString, "images/%04d.png", num++);
		//	
		//	if (ilSaveImage((const ILstring)bufferString) == false)
		//	{
		//		ilDeleteImages(1, &imageID);
		//		return false;
		//	}
		}

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);

	glfwTerminate();
	exit(EXIT_SUCCESS);
}

