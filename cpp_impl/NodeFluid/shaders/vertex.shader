#version 330 core

layout(location = 0) in vec3 i_position;
layout(location = 1) in float i_color_r;
layout(location = 2) in float i_color_g;
layout(location = 3) in float i_color_b;

out float fragDepth;
out vec3 fragColor;

uniform mat4 worldMat, viewMat, projMat;

void main()
{
	gl_Position = projMat * viewMat * worldMat * vec4(i_position, 1.0f);
	fragDepth = gl_Position.z / gl_Position.w;
	gl_PointSize = (1.0 - fragDepth) * 50.0;
	fragColor = vec3(i_color_r, i_color_g, i_color_b);
}
