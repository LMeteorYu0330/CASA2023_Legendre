#version 330 core

in float fragDepth;
in vec3 fragColor;

layout(location = 0) out vec4 vFragColor;

void main()
{
    vec3 N;
    N.xy = gl_PointCoord - vec2(0.5);
    float mag = dot(N.xy, N.xy);
    if (mag > 0.25)
        discard;
 //   N.z = sqrt(1.0 - mag);

    vFragColor = vec4(fragColor, pow(dot(fragColor, fragColor), 4));
}
