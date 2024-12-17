#pragma once
#include <stdio.h>
#include <stdlib.h>

#include <string.h>

#define strcasecmp _stricmp

#include <math.h>

#include"structures.cuh"
using namespace std;

#define MAX_TRIANGLES 20000
#define MAX_SPHERES 100
#define MAX_LIGHTS 100

char* filename = NULL;

// different display modes
#define MODE_DISPLAY 1
#define MODE_JPEG 2

int mode = MODE_DISPLAY;


//#define M_PI 3.14159265358979323846

// the field of view of the camera
//#define fov 60.0

//double aspectRatio = static_cast<double>(WIDTH) / static_cast<double>(HEIGHT);
//double fovRadian = fov * M_PI / 180.0;
//
//unsigned char  buffer[HEIGHT][WIDTH][3];
unsigned char * buffer;

Triangle triangles[MAX_TRIANGLES];
Sphere spheres[MAX_SPHERES];
Light lights[MAX_LIGHTS];
double ambient_light[3];

int num_triangles = 0;
int num_spheres = 0;
int num_lights = 0;
int number_of_objects;

double generateRandomValue(double min, double max);
void subDividedLights();

void parse_check(const char* expected, char* found);
void parse_doubles(FILE* file, const char* check, double p[3]);
void parse_rad(FILE* file, double* r);
void parse_shi(FILE* file, double* shi);
int loadScene(char* argv);
void save_jpg();

