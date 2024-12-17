#ifndef RAY_TRACING_CUH
#define RAY_TRACING_CUH
#include"structures.cuh"
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

// the field of view of the camera
#define fov 60.0

__device__ Triangle d_triangles[20000];
__device__ Sphere d_spheres[100];
__device__ Light d_lights[100];
__device__ double d_ambient_light[3];

__device__ int d_num_triangles;
__device__ int d_num_spheres;
__device__ int d_num_lights;

 double* d_pixelColors;
 unsigned char* d_buffer;

__global__ void gpu_ray_tracing_kernel(double* d_pixelColors);
__global__ void gpu_ray_tracing_superSampling_kernel(double* d_pixelColors, unsigned char* d_buffer);

__device__ double d_aspectRatio = static_cast<double>(WIDTH) / static_cast<double>(HEIGHT);
__device__ double d_fovRadian = fov * M_PI / 180.0;
//__device__ double d_delta = 2 * d_aspectRatio * tan(d_fovRadian / 2.0) / (WIDTH);

__device__ void ray_tracing(double origin[3], double direction[3], int depth, double (& colorFinal)[3]);
//__device__ double* ray_tracing(double origin[3], double direction[3], int depth);
__device__ void generate_ray(double direction[3], int x, int y);
__device__ void generate_ray_super_sampling(double direction[3], int x, int y, int i, int j);
__device__ void cast_shadow_ray(double origin[3], double light[3], double direction[3]);
__device__ double get_color_distance(double c1[3], double c2[3]);
__device__ double sphere_intersection(double t, double origin[3], double direction[3], int& indexSphere);
__device__ double triangle_intersection(double t, double origin[3], double direction[3], double baryCentric[3], int& index);


__device__ void unitCrossProduct(double vectorOne[3], double vectorTwo[3], double result[3]);
__device__ double dotProduct(double vectorOne[3], double vectorTwo[3]);
__device__ double getNorm(double vector[3]);
__device__ void normalize(double vector[3]);

__device__ double square(double x);
__device__ double compute2DTriangleAreaXY(double v1[3], double v2[3], double v3[3]);
__device__ double compute2DTriangleAreaYZ(double v1[3], double v2[3], double v3[3]);
__device__ double compute2DTriangleAreaXZ(double v1[3], double v2[3], double v3[3]);
__device__ void computeReflectVector(double normal[3], double lightVector[3], double result[3]);
__device__ void interpolate(double baryCentric[3], double v0[3], double v1[3], double v2[3], double result[3]);
__device__ void computePhongShading(double lightColor[3], double normal[3], double lightDirectionVector[3], double cameraVector[3], double reflectVector[3], double color_diffuse[3], double color_specular[3], double shininess, double colorFinal[3]);
#endif