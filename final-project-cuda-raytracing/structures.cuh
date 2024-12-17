
#ifndef MY_STRUCT_H
#define MY_STRUCT_H
#define WIDTH 1024
#define HEIGHT 768
struct Vertex
{
	double position[3];
	double color_diffuse[3];
	double color_specular[3];
	double normal[3];
	double shininess;
};

struct Triangle
{
	Vertex v[3];
};

struct Sphere
{
	double position[3];
	double color_diffuse[3];
	double color_specular[3];
	double shininess;
	double radius;
};

struct Light
{
	double position[3];
	double color[3];
};

#endif 