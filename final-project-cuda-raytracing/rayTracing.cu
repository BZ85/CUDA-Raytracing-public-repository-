/* **************************
 * USC CSCI 596 Final Project -- GPU Ray Tracing by CUDA
 * Code based on my own version of USC CSCI 420 Assignment 3 CPU Raytracing
 * Author Name: Xinjie Zhu
 * *************************
*/

/* 11.3.2024 unsolved minor problem: 
  1. recursion depth cannot larger than 2
  2. super sampling part seems slow due to unbalanced work belong threads
*/

/* 11.30.2024 updated:
  Recursion depth problem (cannot larger than 2) is solved by set the stack size for each thread larger
  by cudaDeviceSetLimit
*/
#include <cuda_runtime.h>
#include "rayTracing.cuh"
#include "rayTracing.h"
#include <random>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <iostream>
#include <chrono>


__global__ void gpu_ray_tracing_kernel(double * d_pixelColors) {
    
    double origin[3]{};
    double direction[3]{};
   
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

   
     if (x >= WIDTH || y >= HEIGHT) return;
    // printf("pixel %d %d\n", x, y);
    // 
        // first make an ordinary ray tracing, not use super sampling, for all pixels
    
       double colorFinal[3] = { 1.0, 1.0 ,1.0 };
        generate_ray(direction, x, y);
       // colorFinal = ray_tracing(origin, direction, 1, colorFinal);
        ray_tracing(origin, direction, 4, colorFinal);

      
       
        d_pixelColors[3 * (x + y * WIDTH) + 0] = colorFinal[0] ;
        d_pixelColors[3 * (x + y * WIDTH) + 1] = colorFinal[1] ;
        d_pixelColors[3 * (x + y * WIDTH) + 2] = colorFinal[2] ;
        
}

__global__ void gpu_ray_tracing_superSampling_kernel(double* d_pixelColors, unsigned char* d_buffer) {
    
    double origin[3]{};
    double direction[3]{};
    
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= WIDTH || y >= HEIGHT) return;

    double colorDifference = 2.0;
    
    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1) {
        colorDifference = get_color_distance(&d_pixelColors[3 * (x + (y - 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x + 1 + (y - 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x - 1 + y * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x + 1 + (y + 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x + 1 + y * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x - 1 + (y - 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x - 1 + (y + 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x + (y + 1) * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference += get_color_distance(&d_pixelColors[3 * (x + y * WIDTH)], &d_pixelColors[3 * (x + y * WIDTH)]);
        colorDifference /= 9.0;
    }

    // if difference is big enough, then apply super sampling, otherwise skip this
           // and draw pixel directly, based on former computed rgb value
    //  put the skipping situation in if branch, not in else branch, to enhance performance

    if (colorDifference < 0.1) {
        d_buffer[3 * (x + y * WIDTH) + 0] = d_pixelColors[3 * (x + y * WIDTH) + 0] * 255;
        d_buffer[3 * (x + y * WIDTH) + 1] = d_pixelColors[3 * (x + y * WIDTH) + 1] * 255;
        d_buffer[3 * (x + y * WIDTH) + 2] = d_pixelColors[3 * (x + y * WIDTH) + 2] * 255;
    }
    
   // if (colorDifference > 0.1) {

    else{

        double colorFinal[3] = { 0.0, 0.0, 0.0 };
        // super sampling
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                generate_ray_super_sampling(direction, x, y, i, j);
                double colorSample[3] = { 1.0, 1.0, 1.0 };
                ray_tracing(origin, direction, 4, colorSample);

                colorFinal[0] += colorSample[0];
                colorFinal[1] += colorSample[1];
                colorFinal[2] += colorSample[2];
              
            }

        colorFinal[0] /= 9.0;
        colorFinal[1] /= 9.0;
        colorFinal[2] /= 9.0;

        d_buffer[3 * (x + y * WIDTH) + 0] = colorFinal[0] * 255;
        d_buffer[3 * (x + y * WIDTH) + 1] = colorFinal[1] * 255;
        d_buffer[3 * (x + y * WIDTH) + 2] = colorFinal[2] * 255;

    }

    /*
    else {

        d_buffer[3 * (x + y * WIDTH) + 0] = d_pixelColors[3 * (x + y * WIDTH) + 0] * 255;
        d_buffer[3 * (x + y * WIDTH) + 1] = d_pixelColors[3 * (x + y * WIDTH) + 1] * 255;
        d_buffer[3 * (x + y * WIDTH) + 2] = d_pixelColors[3 * (x + y * WIDTH) + 2] * 255;
    }
    */
}



void draw_scene()
{
    // set the size of stack of each thread to be larger, for deeper recursion 
    size_t stackSize = 8192;
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    cudaError_t err = cudaGetLastError();

    subDividedLights();
    buffer = new unsigned char[HEIGHT * WIDTH * 3];
    //double* pixelC = new double[3 * HEIGHT * WIDTH];


   // double** h_pixelColors; 
    // record pixel colors for deciding whether apply super sampling or not
    cudaMalloc(&d_pixelColors, 3 * HEIGHT * WIDTH * sizeof(double));
   // cudaMalloc(&h_pixelColors, HEIGHT * WIDTH * sizeof(double*));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error dpixel: %s\n", cudaGetErrorString(err));
    }

    cudaMalloc(&d_buffer, 3 * HEIGHT * WIDTH * sizeof(unsigned char));
 //   cudaMalloc(&d_triangles, num_triangles * sizeof(Triangle));
   // cudaMalloc(&d_spheres, num_spheres * sizeof(Sphere));
   // cudaMalloc(&d_lights, num_lights * sizeof(Light));
   // cudaMalloc(&d_ambient_light, 3 * sizeof(double));



   // cudaMemcpy(d_pixelColors, h_pixelColors, HEIGHT * WIDTH * sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_triangles, triangles, num_triangles * sizeof(Triangle), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles1: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpyToSymbol(d_spheres, spheres, num_spheres * sizeof(Sphere), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles2: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpyToSymbol(d_lights, lights, num_lights * sizeof(Light), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles3: %s\n", cudaGetErrorString(err));
    }


    cudaMemcpyToSymbol(d_ambient_light, ambient_light, 3 * sizeof(double), 0, cudaMemcpyHostToDevice);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles4: %s\n", cudaGetErrorString(err));
    }




    cudaMemcpyToSymbol(d_num_triangles, &num_triangles, sizeof(int), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles5: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpyToSymbol(d_num_spheres, &num_spheres, sizeof(int), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles6: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpyToSymbol(d_num_lights, &num_lights, sizeof(int), 0, cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error copying triangles7: %s\n", cudaGetErrorString(err));
    }
  
    // allocate the block based on the WIDTH and HEIGHT of output image
    dim3 numBlocks(ceil(WIDTH/16), ceil(HEIGHT/16), 1);
    dim3 threads_per_block(16, 16, 1);

    gpu_ray_tracing_kernel <<<numBlocks, threads_per_block >>> (d_pixelColors);
   
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernal One error: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA kernal One successfully running\n");
    }
    
    gpu_ray_tracing_superSampling_kernel << <numBlocks, threads_per_block >> > (d_pixelColors, d_buffer);

        cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernal Two error: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA kernal Two successfully running\n");
    }

    //cudaMemcpy(pixelC, d_pixelColors, 3 * HEIGHT * WIDTH * sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA successfully running\n");
    }

    cudaMemcpy(buffer, d_buffer, 3 * HEIGHT * WIDTH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    else {
        printf("CUDA successfully running\n");
    }
    
    /*
    for (int i = 0; i < 3 * HEIGHT * WIDTH; i++) {
     //    buffer[i] = (unsigned char) pixelC[i];
       // std::cout << buffer[i] << endl;
    }
    */


    save_jpg();

    cudaFree(d_pixelColors);
    cudaFree(d_buffer);
    cudaFree(d_triangles);
    cudaFree(d_spheres);
    cudaFree(d_lights);

    free(buffer);
//    cudaFree(d_num_triangles);
  //  cudaFree(d_num_spheres);
   // cudaFree(d_num_lights);

}


 void subDividedLights() {
    // Light subLights, to create soft shadow
    int numlightsIndex = num_lights;

    // adjust the sublights num based on different num of objects
    int num_sublight_per_source = 30;


    if (number_of_objects > 2) { // do not use sublight if there is only one object and one light

        if (number_of_objects > 5) num_sublight_per_source = 10;
        if (number_of_objects > 300) num_sublight_per_source = 3;
        if (number_of_objects > 1000) num_sublight_per_source = 3;

        for (int i = 0; i < num_lights; i++) {

            // divide the power of original light source
            lights[i].color[0] /= (num_sublight_per_source * 1.0 + 1); // the multiply factor can be change to adjust color if recursive reflecting from more lights
            lights[i].color[1] /= (num_sublight_per_source * 1.0 + 1); // the factor is 1.0 if using reflectSum
            lights[i].color[2] /= (num_sublight_per_source * 1.0 + 1);

            for (int j = 0; j < num_sublight_per_source; j++) {

                Light sublight{};

                // assign the power from original light source to sublights
                sublight.color[0] = lights[i].color[0];
                sublight.color[1] = lights[i].color[1];
                sublight.color[2] = lights[i].color[2];

                sublight.position[0] = lights[i].position[0] + generateRandomValue(0.0, 0.1); // the range of sublight's position may change for different scenes
                sublight.position[1] = lights[i].position[1] + generateRandomValue(0.0, 0.1);
                sublight.position[2] = lights[i].position[2] + generateRandomValue(0.0, 0.1);

                lights[numlightsIndex++] = sublight;


            }

        }

        num_lights = numlightsIndex;
    }
}


 __device__ void generate_ray(double direction[3], int x, int y) {

     double d_delta = 2 * d_aspectRatio * tan(d_fovRadian / 2.0) / (WIDTH);

     direction[0] = -1 * d_aspectRatio * tan(d_fovRadian / 2) + x * d_delta;
     direction[1] = -1 * tan(d_fovRadian / 2) + y * d_delta;
     direction[2] = -1;

   
      // normalize the ray direction
     double length = sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);

     if (length != 0) {
         direction[0] = direction[0] / length;
         direction[1] = direction[1] / length;
         direction[2] = direction[2] / length;
     }

 }

 __device__ void generate_ray_super_sampling(double direction[3], int x, int y, int i, int j) {

     double d_delta = 2 * d_aspectRatio * tan(d_fovRadian / 2.0) / (WIDTH);

     direction[0] = -1 * d_aspectRatio * tan(d_fovRadian / 2) + x * d_delta + i * d_delta / 3.0;
     direction[1] = -1 * tan(d_fovRadian / 2) + y * d_delta + +j * d_delta / 3.0;
     direction[2] = -1;

    
     double length = sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
     
     if (length != 0) {
         direction[0] = direction[0] / length;
         direction[1] = direction[1] / length;
         direction[2] = direction[2] / length;
     }

 }

 __device__ void ray_tracing(double origin[3], double direction[3], int depth, double (& colorFinal)[3]) {

     // cout << depth << endl;
     //if (depth <= 0) return nullptr;
     if (depth <= 0) return;
     double tDefault = 100000000.0; // the default t value and never changes, use to compare with new computed t

     double tFinal;
     double baryCentric[3]{};
     int indexTri = 0, indexSphere = 0;
     double tIntersectionSphere = sphere_intersection(tDefault, origin, direction, indexSphere);

     double tIntersectionTriangle = triangle_intersection(tDefault, origin, direction, baryCentric, indexTri);

     tFinal = min(tIntersectionSphere, tIntersectionTriangle);
   
     if (tFinal != tDefault) {// ray is intersected with at least one object, and this object is the first one that intersect with
         
         double surfacePoint[3] = { origin[0] + tFinal * direction[0], origin[1] + tFinal * direction[1], origin[2] + tFinal * direction[2] };
         double shadowRayDirection[3]{};

       //  double* colorFinal = new double[3]; // only set dynamic memory can return by the function
        // double colorFinal[3]{};
        // cudaMalloc(&colorFinal, 3 * sizeof(double));
     
         colorFinal[0] = d_ambient_light[0];
         colorFinal[1] = d_ambient_light[1];
         colorFinal[2] = d_ambient_light[2];
         double dumb[3]{};
         int dumbIndex = 0;

         double recursiveReflectionVector[3]{};
         
         
         if (tFinal == tIntersectionSphere) { // intersect with sphere

             for (int i = 0; i < d_num_lights; i++) { // for every light source
                 cast_shadow_ray(surfacePoint, d_lights[i].position, shadowRayDirection);
                 tIntersectionSphere = sphere_intersection(tDefault, surfacePoint, shadowRayDirection, dumbIndex);

                 // we don't need barycentric coordinate and triangle index here for shadow block detecting, so I use dumb variables to be parameters
                 tIntersectionTriangle = triangle_intersection(tDefault, surfacePoint, shadowRayDirection, dumb, dumbIndex);
                 tFinal = min(tIntersectionSphere, tIntersectionTriangle);

                 // compute the distance from surface point to light source
                 double lightDistanceVector[3] = { d_lights[i].position[0] - surfacePoint[0], d_lights[i].position[1] - surfacePoint[1] ,d_lights[i].position[2] - surfacePoint[2] };
                 double lightDistance = getNorm(lightDistanceVector);
                 double distancePQ = 0;

                 double normalSphere[3] = { surfacePoint[0] - d_spheres[indexSphere].position[0], surfacePoint[1] - d_spheres[indexSphere].position[1], surfacePoint[2] - d_spheres[indexSphere].position[2] };
                 normalize(normalSphere);

                 double eyeDirection[3]{};
                 eyeDirection[0] = direction[0] * -1.0;
                 eyeDirection[1] = direction[1] * -1.0;
                 eyeDirection[2] = direction[2] * -1.0;


                 /* VERY IMPORTANT, updated 4.23.2024
                  there is a difference between recursiveReflection vector and phong reflective vector
                 Recursive Reflection vector is the reflect vector of traced ray (direction)
                 It is not affected by different light source
                 Different light source has same recursive reflection vector

                 Phong reflect vector is the reflect vector of shadow ray

                 Same comment as in the triangle part
                 */

                 
     
                 computeReflectVector(normalSphere, eyeDirection, recursiveReflectionVector);

                 double phongReflectVectors[3]{};

                 computeReflectVector(normalSphere, shadowRayDirection, phongReflectVectors);

                 if (tFinal != tDefault) {

                     double blockPoint[3] = { surfacePoint[0] + tFinal * shadowRayDirection[0], surfacePoint[1] + tFinal * shadowRayDirection[1], surfacePoint[2] + tFinal * shadowRayDirection[2] };
                     double distancePQVector[3] = { blockPoint[0] - surfacePoint[0], blockPoint[1] - surfacePoint[1] , blockPoint[2] - surfacePoint[2] };
                     distancePQ = getNorm(distancePQVector);
                 }

                 if (tFinal == tDefault || distancePQ > lightDistance) { // shadow ray not blocked by any objects, or blocked objects is behind light source

                     // preparing the factors and compute phong shading
                     double cameraVector[3] = { -surfacePoint[0], -surfacePoint[1], -surfacePoint[2] };
                     normalize(cameraVector);

                     double color[3]{};


                     //   computeReflectVector(normalSphere, shadowRayDirection, reflectVector);


                     computePhongShading(d_lights[i].color, normalSphere, shadowRayDirection, cameraVector, phongReflectVectors, d_spheres[indexSphere].color_diffuse, d_spheres[indexSphere].color_specular, d_spheres[indexSphere].shininess, color);

                     colorFinal[0] += color[0];
                     colorFinal[1] += color[1];
                     colorFinal[2] += color[2];

                 }


             }

         }


         else { // intersect with triangle


             for (int i = 0; i < d_num_lights; i++) { // for every light source
                 cast_shadow_ray(surfacePoint, d_lights[i].position, shadowRayDirection);
                 tIntersectionSphere = sphere_intersection(tDefault, surfacePoint, shadowRayDirection, dumbIndex);

                 // we don't need barycentric coordinate and triangle index here for shadow block detecting, so I use dumb variables to be parameters
                 tIntersectionTriangle = triangle_intersection(tDefault, surfacePoint, shadowRayDirection, dumb, dumbIndex);
                 tFinal = min(tIntersectionSphere, tIntersectionTriangle);

                 // compute the distance from surface point to light source
                 double lightDistanceVector[3] = { d_lights[i].position[0] - surfacePoint[0], d_lights[i].position[1] - surfacePoint[1] ,d_lights[i].position[2] - surfacePoint[2] };
                 double lightDistance = getNorm(lightDistanceVector);
                 double distancePQ = 0;
                 double normalFinal[3]{};

                 // compute normal and reflect vector, even for the shadow ray which is blocked
                 // because we need to do recursive reflection for blocked shadow ray
                 // even if the shadow ray is blocked, its reflected ray is not blocked
                 interpolate(baryCentric, d_triangles[indexTri].v[0].normal, d_triangles[indexTri].v[1].normal, d_triangles[indexTri].v[2].normal, normalFinal);
                 normalize(normalFinal);

                 double eyeDirection[3]{};
                 eyeDirection[0] = direction[0] * -1.0;
                 eyeDirection[1] = direction[1] * -1.0;
                 eyeDirection[2] = direction[2] * -1.0;
                 computeReflectVector(normalFinal, eyeDirection, recursiveReflectionVector);

                 double phongReflectVectors[3]{};

                 computeReflectVector(normalFinal, shadowRayDirection, phongReflectVectors);

                 if (tFinal != tDefault) {

                     double blockPoint[3] = { surfacePoint[0] + tFinal * shadowRayDirection[0], surfacePoint[1] + tFinal * shadowRayDirection[1], surfacePoint[2] + tFinal * shadowRayDirection[2] };
                     double distancePQVector[3] = { blockPoint[0] - surfacePoint[0], blockPoint[1] - surfacePoint[1] , blockPoint[2] - surfacePoint[2] };
                     distancePQ = getNorm(distancePQVector);
                 }

                 if (tFinal == tDefault || distancePQ > lightDistance) { // shadow ray not blocked by any objects, or blocked objects is behind light source
                     // preparing the factors and compute phong shading
                     double  color_diffuse_final[3]{}, color_specular_final[3]{};
                     double shininess = 0;
                     double cameraVector[3] = { -surfacePoint[0], -surfacePoint[1], -surfacePoint[2] };

                     double color[3]{};

                     normalize(cameraVector);

                     interpolate(baryCentric, d_triangles[indexTri].v[0].color_diffuse, d_triangles[indexTri].v[1].color_diffuse, d_triangles[indexTri].v[2].color_diffuse, color_diffuse_final);
                     interpolate(baryCentric, d_triangles[indexTri].v[0].color_specular, d_triangles[indexTri].v[1].color_specular, d_triangles[indexTri].v[2].color_specular, color_specular_final);



                     shininess = baryCentric[0] * d_triangles[indexTri].v[0].shininess + baryCentric[1] * d_triangles[indexTri].v[1].shininess + baryCentric[2] * d_triangles[indexTri].v[2].shininess;

                     computePhongShading(d_lights[i].color, normalFinal, shadowRayDirection, cameraVector, phongReflectVectors, color_diffuse_final, color_specular_final, shininess, color);

                     colorFinal[0] += color[0];
                     colorFinal[1] += color[1];
                     colorFinal[2] += color[2];

                 }

             }


         }
         // if rgb is bigger than 1.0, clamp to 1.0
         if (colorFinal[0] > 1.0) colorFinal[0] = 1.0;
         if (colorFinal[1] > 1.0) colorFinal[1] = 1.0;
         if (colorFinal[2] > 1.0) colorFinal[2] = 1.0;

         depth--;


       //  double* reflectColor = nullptr;
         double reflectColor[3] = {1.0, 1.0 ,1.0};
         if (depth > 0)  ray_tracing(surfacePoint, recursiveReflectionVector, depth, reflectColor);

        

         if (depth > 0) { // if depth = 0, then do not compute color again

             double attenuate = 0.1;

             colorFinal[0] += attenuate * reflectColor[0];
             colorFinal[1] += attenuate * reflectColor[1];
             colorFinal[2] += attenuate * reflectColor[2];

         }
         // if rgb is bigger than 1.0, clamp to 1.0
         if (colorFinal[0] > 1.0) colorFinal[0] = 1.0;
         if (colorFinal[1] > 1.0) colorFinal[1] = 1.0;
         if (colorFinal[2] > 1.0) colorFinal[2] = 1.0;
         //delete[] reflectColor;


         //cudaFree(reflectColor);
         
        // return colorFinal;

         
        // return nullptr;
     }

   //  else return nullptr; // if no intersection from the screen pixel to any objects, then do not return color
   // else colorFinal = nullptr;

 }
 __device__ void cast_shadow_ray(double origin[3], double light[3], double direction[3]) {

     direction[0] = light[0] - origin[0];
     direction[1] = light[1] - origin[1];
     direction[2] = light[2] - origin[2];

     // normalize the ray direction
     double length = sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);

     if (length != 0) {
         direction[0] = direction[0] / length;
         direction[1] = direction[1] / length;
         direction[2] = direction[2] / length;
     }

 }

 // compare the current color pixels to three generated neigbor
 __device__ double get_color_distance(double c1[3], double c2[3]) {

     return sqrt(square(c1[0] - c2[0]) + square(c1[1] - c2[1]) + square(c1[2] - c2[2]));

 }

 __device__ double sphere_intersection(double t, double origin[3], double direction[3], int& indexSphere) {
     for (int i = 0; i < d_num_spheres; i++) {
         int len = 3;

         double x0, y0, z0, xc, yc, zc, xd, yd, zd, r, a, b, c;
         double t0 = 0, t1 = 0;
         x0 = origin[0];
         y0 = origin[1];
         z0 = origin[2];
         xd = direction[0];
         yd = direction[1];
         zd = direction[2];
         xc = d_spheres[i].position[0];
         yc = d_spheres[i].position[1];
         zc = d_spheres[i].position[2];
         r = d_spheres[i].radius;
         a = square(xd) + square(yd) + square(zd);
         b = 2 * (xd * (x0 - xc) + yd * (y0 - yc) + zd * (z0 - zc));
         c = square(x0 - xc) + square(y0 - yc) + square(z0 - zc) - square(r);

         double delta = b * b - 4 * a * c;
         //   cout << delta << endl;
         if (delta >= 0) {

             t0 = (-b + sqrt(delta)) / (2 * a);
             t1 = (-b - sqrt(delta)) / (2 * a);

             //   cout << t0 << " " << t1 << endl;
             if (t0 > 0.001 && t1 > 0.001) {
                 if (t > min(t0, t1)) {
                     t = min(t0, t1);
                     indexSphere = i;  // record the index of intersected sphere

                 }
             }
         }


     }

     return t;
 }

 __device__ double triangle_intersection(double t, double origin[3], double direction[3], double baryCentric[3], int& index) { // t is current smallest t
     for (int i = 0; i < d_num_triangles; i++) {
         double normal[3]{}; // triangle normal
         double vectorOne[3]{};
         double vectorTwo[3]{};
         double d; // d is the coefficient of plane , ax + by + cz + d = 0
         double t0;
         double dotProductNP0;
         double dotProductND;
         vectorOne[0] = d_triangles[i].v[0].position[0] - d_triangles[i].v[1].position[0];
         vectorOne[1] = d_triangles[i].v[0].position[1] - d_triangles[i].v[1].position[1];
         vectorOne[2] = d_triangles[i].v[0].position[2] - d_triangles[i].v[1].position[2];

         vectorTwo[0] = d_triangles[i].v[2].position[0] - d_triangles[i].v[0].position[0];
         vectorTwo[1] = d_triangles[i].v[2].position[1] - d_triangles[i].v[0].position[1];
         vectorTwo[2] = d_triangles[i].v[2].position[2] - d_triangles[i].v[0].position[2];

         unitCrossProduct(vectorOne, vectorTwo, normal);

         // d = -ax - by - cz
         d = -normal[0] * d_triangles[i].v[0].position[0] - normal[1] * d_triangles[i].v[0].position[1] - normal[2] * d_triangles[i].v[0].position[2];

         dotProductNP0 = dotProduct(normal, origin);
         dotProductND = dotProduct(normal, direction);
         if (dotProductND != 0) {// if intersect with plane
             t0 = -(dotProductNP0 + d) / dotProductND;
             if (t0 > 0.001) { // if intersection is front of the ray origin
                 double intersectPoint[3] = {};
                 intersectPoint[0] = origin[0] + t0 * direction[0];
                 intersectPoint[1] = origin[1] + t0 * direction[1];
                 intersectPoint[2] = origin[2] + t0 * direction[2];

                 double xyNormal[3] = { 0, 0, 1 };
                 double yzNormal[3] = { 1, 0, 0 };
                 double xzNormal[3] = { 0, 1 ,0 };

                 double dotProductXY = abs(dotProduct(normal, xyNormal));
                 double dotProductYZ = abs(dotProduct(normal, yzNormal));
                 double dotProductXZ = abs(dotProduct(normal, xzNormal));

                 double tri012, trip12, tri0p2, tri01p;
                 double alpha, beta, gama; // barycentric coordinate
                 // project onto xy plane
                 if (dotProductXY == max(dotProductXY, max(dotProductYZ, dotProductXZ))) {
                     tri012 = compute2DTriangleAreaXY(d_triangles[i].v[0].position, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     trip12 = compute2DTriangleAreaXY(intersectPoint, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     tri0p2 = compute2DTriangleAreaXY(d_triangles[i].v[0].position, intersectPoint, d_triangles[i].v[2].position);
                     tri01p = compute2DTriangleAreaXY(d_triangles[i].v[0].position, d_triangles[i].v[1].position, intersectPoint);

                 }
                 // project onto yz plane
                 else if (dotProductYZ == max(dotProductXY, max(dotProductYZ, dotProductXZ))) {
                     tri012 = compute2DTriangleAreaYZ(d_triangles[i].v[0].position, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     trip12 = compute2DTriangleAreaYZ(intersectPoint, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     tri0p2 = compute2DTriangleAreaYZ(d_triangles[i].v[0].position, intersectPoint, d_triangles[i].v[2].position);
                     tri01p = compute2DTriangleAreaYZ(d_triangles[i].v[0].position, d_triangles[i].v[1].position, intersectPoint);
                 }
                 // project onto xz plane
                 else if (dotProductXZ == max(dotProductXY, max(dotProductYZ, dotProductXZ))) {
                     tri012 = compute2DTriangleAreaXZ(d_triangles[i].v[0].position, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     trip12 = compute2DTriangleAreaXZ(intersectPoint, d_triangles[i].v[1].position, d_triangles[i].v[2].position);
                     tri0p2 = compute2DTriangleAreaXZ(d_triangles[i].v[0].position, intersectPoint, d_triangles[i].v[2].position);
                     tri01p = compute2DTriangleAreaXZ(d_triangles[i].v[0].position, d_triangles[i].v[1].position, intersectPoint);
                 }

                 alpha = trip12 / tri012;
                 beta = tri0p2 / tri012;
                 gama = tri01p / tri012;



                 if (alpha > 0 && beta > 0 && gama > 0) { // if all bigger than 0, then point is inside triangle
                     if (t > t0) {
                         t = t0;
                         index = i; // record the index of triangle
                         baryCentric[0] = alpha;
                         baryCentric[1] = beta;
                         baryCentric[2] = gama;
                     }
                 }

             }
         }
     }

     return t;


 }


__device__ void computeReflectVector(double normal[3], double lightVector[3], double result[3]) { // lightVector is the direction from surface point to light source
    double dotP = dotProduct(normal, lightVector);


    result[0] = 2 * dotP * normal[0] - lightVector[0];
    result[1] = 2 * dotP * normal[1] - lightVector[1];
    result[2] = 2 * dotP * normal[2] - lightVector[2];


    if (dotP < 0) {
        result[0] = lightVector[0];
        result[1] = lightVector[1];
        result[2] = lightVector[2];
    }

    // result is already unit vector
}

__device__ void computePhongShading(double lightColor[3], double normal[3], double lightDirectionVector[3], double cameraVector[3], double reflectVector[3], double color_diffuse[3], double color_specular[3], double shininess, double colorFinal[3]) {

    double lDotN = dotProduct(lightDirectionVector, normal);
    double rDotV = dotProduct(reflectVector, cameraVector);

    // if dot product is smaller than 0, then clamp it to 0
    if (lDotN < 0.0) lDotN = 0.0;
    if (rDotV < 0.0) rDotV = 0.0;

    colorFinal[0] = lightColor[0] * (color_diffuse[0] * lDotN + color_specular[0] * pow(rDotV, shininess));
    colorFinal[1] = lightColor[1] * (color_diffuse[1] * lDotN + color_specular[1] * pow(rDotV, shininess));
    colorFinal[2] = lightColor[2] * (color_diffuse[2] * lDotN + color_specular[2] * pow(rDotV, shininess));


}
__device__ void unitCrossProduct(double vectorOne[3], double vectorTwo[3], double result[3]) {

    result[0] = vectorOne[1] * vectorTwo[2] - vectorOne[2] * vectorTwo[1];
    result[1] = vectorOne[2] * vectorTwo[0] - vectorOne[0] * vectorTwo[2];
    result[2] = vectorOne[0] * vectorTwo[1] - vectorOne[1] * vectorTwo[0];

    double length = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);

    if (length != 0) { // if length = 0, then do not nomralize it, otherwise the value will be nan
        result[0] = result[0] / length;
        result[1] = result[1] / length;
        result[2] = result[2] / length;
    }

}

__device__ void interpolate(double baryCentric[3], double v0[3], double v1[3], double v2[3], double result[3]) {

    result[0] = baryCentric[0] * v0[0] + baryCentric[1] * v1[0] + baryCentric[2] * v2[0];
    result[1] = baryCentric[0] * v0[1] + baryCentric[1] * v1[1] + baryCentric[2] * v2[1];
    result[2] = baryCentric[0] * v0[2] + baryCentric[1] * v1[2] + baryCentric[2] * v2[2];

}
__device__ double dotProduct(double vectorOne[3], double vectorTwo[3]) {
    return vectorOne[0] * vectorTwo[0] + vectorOne[1] * vectorTwo[1] + vectorOne[2] * vectorTwo[2];
}

__device__ double getNorm(double vector[3]) {
    return sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
}

__device__ void normalize(double vector[3]) {
    double length = getNorm(vector);
    if (length != 0) {
        vector[0] = vector[0] / length;
        vector[1] = vector[1] / length;
        vector[2] = vector[2] / length;
    }
}

__device__ double compute2DTriangleAreaXY(double v1[3], double v2[3], double v3[3]) { // specific orientation for v1, v2, v3
    return (1.0 / 2) * ((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]));
}

__device__ double compute2DTriangleAreaYZ(double v1[3], double v2[3], double v3[3]) { // specific orientation for v1, v2, v3
    return (1.0 / 2) * ((v2[1] - v1[1]) * (v3[2] - v1[2]) - (v3[1] - v1[1]) * (v2[2] - v1[2]));
}

__device__ double compute2DTriangleAreaXZ(double v1[3], double v2[3], double v3[3]) { // specific orientation for v1, v2, v3
    return (1.0 / 2) * ((v2[0] - v1[0]) * (v3[2] - v1[2]) - (v3[0] - v1[0]) * (v2[2] - v1[2]));
}

__device__ double square(double x) {
    return x * x;
}

double generateRandomValue(double min, double max) {

    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<double> dis(min, max);

    double randomNumber = dis(gen);
    return randomNumber;

}

void parse_check(const char* expected, char* found)
{
    if (strcasecmp(expected, found))
    {
        printf("Expected '%s ' found '%s '\n", expected, found);
        printf("Parse error, abnormal abortion\n");
        exit(0);
    }
}

void parse_doubles(FILE* file, const char* check, double p[3])
{
    char str[100];
    fscanf(file, "%s", str);
    parse_check(check, str);
    fscanf(file, "%lf %lf %lf", &p[0], &p[1], &p[2]);
    printf("%s %lf %lf %lf\n", check, p[0], p[1], p[2]);
}

void parse_rad(FILE* file, double* r)
{
    char str[100];
    fscanf(file, "%s", str);
    parse_check("rad:", str);
    fscanf(file, "%lf", r);
    printf("rad: %f\n", *r);
}

void parse_shi(FILE* file, double* shi)
{
    char s[100];
    fscanf(file, "%s", s);
    parse_check("shi:", s);
    fscanf(file, "%lf", shi);
    printf("shi: %f\n", *shi);
}

int loadScene(char* argv)
{
    FILE* file = fopen(argv, "r");
    // int number_of_objects;
    char type[50];
    Triangle t;
    Sphere s;
    Light l;
    fscanf(file, "%i", &number_of_objects);

    printf("number of objects: %i\n", number_of_objects);

    parse_doubles(file, "amb:", ambient_light);

    for (int i = 0; i < number_of_objects; i++)
    {
        fscanf(file, "%s\n", type);
        printf("%s\n", type);
        if (strcasecmp(type, "triangle") == 0)
        {
            printf("found triangle\n");
            for (int j = 0;j < 3;j++)
            {
                parse_doubles(file, "pos:", t.v[j].position);
                parse_doubles(file, "nor:", t.v[j].normal);
                parse_doubles(file, "dif:", t.v[j].color_diffuse);
                parse_doubles(file, "spe:", t.v[j].color_specular);
                parse_shi(file, &t.v[j].shininess);

            }

            if (num_triangles == MAX_TRIANGLES)
            {
                printf("too many triangles, you should increase MAX_TRIANGLES!\n");
                exit(0);
            }
            triangles[num_triangles++] = t;
        }
        else if (strcasecmp(type, "sphere") == 0)
        {
            printf("found sphere\n");

            parse_doubles(file, "pos:", s.position);
            parse_rad(file, &s.radius);
            parse_doubles(file, "dif:", s.color_diffuse);
            parse_doubles(file, "spe:", s.color_specular);
            parse_shi(file, &s.shininess);

            if (num_spheres == MAX_SPHERES)
            {
                printf("too many spheres, you should increase MAX_SPHERES!\n");
                exit(0);
            }
            spheres[num_spheres++] = s;
        }
        else if (strcasecmp(type, "light") == 0)
        {
            printf("found light\n");
            parse_doubles(file, "pos:", l.position);
            parse_doubles(file, "col:", l.color);

            if (num_lights == MAX_LIGHTS)
            {
                printf("too many lights, you should increase MAX_LIGHTS!\n");
                exit(0);
            }
            lights[num_lights++] = l;
        }
        else
        {
            printf("unknown type in scene description:\n%s\n", type);
            exit(0);
        }
    }
    return 0;
}

/*
void save_jpg()
{
    printf("Saving JPEG file: %s\n", filename);

    ImageIO img(WIDTH, HEIGHT, 3, buffer);
    if (img.save(filename, ImageIO::FORMAT_JPEG) != ImageIO::OK)
        printf("Error in Saving\n");
    else
        printf("File saved Successfully\n");
}
*/


void save_jpg() {
    
    
   // stbi_write_jpg(filename, WIDTH, HEIGHT, 3, buffer, 300);

    // flip the picture to the normal status according to stbi's requirement
    unsigned char* buffer_reversed = new unsigned char[WIDTH * HEIGHT * 3];

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            int src_index = (y * WIDTH + x) * 3;  
            int dst_index = ((HEIGHT - 1 - y) * WIDTH + x) * 3;  

            buffer_reversed[dst_index] = buffer[src_index];      
            buffer_reversed[dst_index + 1] = buffer[src_index + 1];  
            buffer_reversed[dst_index + 2] = buffer[src_index + 2]; 
        }
    }

    // then save it
    stbi_write_jpg(filename, WIDTH, HEIGHT, 3, buffer_reversed, 300);

    delete[] buffer_reversed; 

}

int main(int argc, char** argv) {

    loadScene(argv[1]);
    filename = argv[2];

    auto startTime = std::chrono::high_resolution_clock::now();

    draw_scene();

    auto endTime = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = endTime - startTime;
    std::cout << "Running time: " << duration.count() << std::endl;

    return 0;
}