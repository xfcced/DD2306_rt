#include <iostream>
#include <fstream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include <cstring>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
// Linear traversal color function
__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

// BVH traversal color function
__device__ vec3 color_bvh(const ray& r, BVHNode *bvh_nodes, hitable **d_list, curandState *local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if (bvh_hit(bvh_nodes, d_list, cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

// Export sphere geometry data from GPU to CPU (for BVH construction)
__global__ void export_sphere_data(hitable **d_list, SphereGeom *geom_data, int num_spheres) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for(int i = 0; i < num_spheres; i++) {
            sphere *s = (sphere *)d_list[i];
            geom_data[i].center = s->center;
            geom_data[i].radius = s->radius;
            geom_data[i].list_idx = i;
        }
    }
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, bool use_bvh, BVHNode *bvh_nodes, hitable **d_list) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        
        // Choose traversal method based on flag
        if (use_bvh) {
            col += color_bvh(r, bvh_nodes, d_list, &local_rand_state);
        } else {
            col += color(r, world, &local_rand_state);
        }
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state, int num_small_spheres, int *actual_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;
        d_list[0] = new sphere(vec3(0,-1000.0,-1), 1000,
                               new lambertian(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        // Calculate grid size based on number of small spheres
        // Use a slightly larger grid to ensure we can create enough spheres
        int grid_size = (int)ceil(sqrt((float)num_small_spheres));
        int half_grid = (grid_size + 1) / 2;  // Round up for odd sizes
        
        for(int a = -half_grid; a <= half_grid && i <= num_small_spheres; a++) {
            for(int b = -half_grid; b <= half_grid && i <= num_small_spheres; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new lambertian(vec3(RND*RND, RND*RND, RND*RND)));
                }
                else if(choose_mat < 0.95f) {
                    d_list[i++] = new sphere(center, 0.2,
                                             new metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)), 0.5f*RND));
                }
                else {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        // Record actual number of small spheres created
        int actual_small_spheres = i - 1;
        
        d_list[i++] = new sphere(vec3(0, 1,0),  1.0, new dielectric(1.5));
        d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = new sphere(vec3(4, 1, 0),  1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world  = new hitable_list(d_list, i);  // Use actual count
        *actual_count = i;  // Return actual total count

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0; (lookfrom-lookat).length();
        float aperture = 0.1;
        *d_camera   = new camera(lookfrom,
                                 lookat,
                                 vec3(0,1,0),
                                 30.0,
                                 float(nx)/float(ny),
                                 aperture,
                                 dist_to_focus);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera, int num_hitables) {
    for(int i=0; i < num_hitables; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

enum MemoryMode {
    MEM_EXPLICIT = 0,  // cudaMalloc + explicit cudaMemcpy
    MEM_UM = 1,        // Unified Memory without hints
    MEM_UM_PREFETCH = 2,  // Unified Memory + cudaMemPrefetchAsync
    MEM_UM_ADVISE = 3     // Unified Memory + cudaMemAdvise
};

int main(int argc, char** argv) {
    int nx = 1200;
    int ny = 800;
    int ns = 20;
    int tx = 16;
    int ty = 16;
    int num_small_spheres = 22*22;  // Default: 484 small spheres
    
    // Parse command line arguments
    bool use_bvh = false;
    MemoryMode mem_mode = MEM_EXPLICIT;
    for(int i = 1; i < argc; i++) {
        if(strcmp(argv[i], "--use-bvh") == 0 || strcmp(argv[i], "-b") == 0) {
            use_bvh = true;
        }
        else if((strcmp(argv[i], "--samples") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            ns = atoi(argv[++i]);
            if(ns <= 0) ns = 20;  // Fallback to default if invalid
        }
        else if((strcmp(argv[i], "--mem-mode") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            int mode = atoi(argv[++i]);
            if(mode >= 0 && mode <= 3) {
                mem_mode = (MemoryMode)mode;
            }
        }
        else if((strcmp(argv[i], "--objects") == 0 || strcmp(argv[i], "-o") == 0) && i + 1 < argc) {
            num_small_spheres = atoi(argv[++i]);
            if(num_small_spheres <= 0) num_small_spheres = 22*22;  // Fallback to default if invalid
        }
    }

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    std::cerr << "Traversal method: " << (use_bvh ? "BVH" : "Linear") << "\n";
    const char* mem_mode_names[] = {"Explicit", "UM", "UM+Prefetch", "UM+Advise"};
    std::cerr << "Memory mode: " << mem_mode_names[mem_mode] << "\n";
    std::cerr << "Number of objects: " << (num_small_spheres + 1 + 3) << " (" << num_small_spheres << " small + 1 ground + 3 large)\n";

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(vec3);

    clock_t scene_start = clock(); // scene creation timer start

    // allocate FB based on memory mode
    vec3 *fb;
    vec3 *h_fb = nullptr;  // Host buffer for explicit mode
    if (mem_mode == MEM_EXPLICIT) {
        // Explicit: allocate device memory and host buffer
        checkCudaErrors(cudaMalloc((void **)&fb, fb_size));
        h_fb = new vec3[num_pixels];
    } else {
        // UM modes: use unified memory
        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));
        if (mem_mode == MEM_UM_PREFETCH) {
            checkCudaErrors(cudaMemPrefetchAsync(fb, fb_size, 0));
        } else if (mem_mode == MEM_UM_ADVISE) {
            checkCudaErrors(cudaMemAdvise(fb, fb_size, cudaMemAdviseSetPreferredLocation, 0));
            checkCudaErrors(cudaMemAdvise(fb, fb_size, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
        }
    }

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1,1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    hitable **d_list;
    int max_hitables = num_small_spheres+1+3;  // Maximum possible: small spheres + ground + 3 large spheres
    checkCudaErrors(cudaMalloc((void **)&d_list, max_hitables*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    
    // Allocate space for actual count
    int *d_actual_count;
    checkCudaErrors(cudaMalloc((void **)&d_actual_count, sizeof(int)));
    
    create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, num_small_spheres, d_actual_count);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Get actual count from device
    int num_hitables;
    checkCudaErrors(cudaMemcpy(&num_hitables, d_actual_count, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_actual_count));
    
    clock_t scene_stop = clock();
    double scene_time = ((double)(scene_stop - scene_start)) / CLOCKS_PER_SEC * 1000.0;
    std::cerr << "Scene creation took " << scene_time << " ms.\n";
    
    // Build BVH if requested
    BVHNode *d_bvh = nullptr;
    int num_bvh_nodes = 0;
    if (use_bvh) {
        clock_t bvh_start = clock(); // BVH building timer start
        
        SphereGeom *geom;
        SphereGeom *h_geom = nullptr;
        
        if (mem_mode == MEM_EXPLICIT) {
            // Explicit mode: device memory + explicit copy
            checkCudaErrors(cudaMalloc((void **)&geom, num_hitables*sizeof(SphereGeom)));
            export_sphere_data<<<1,1>>>(d_list, geom, num_hitables);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            
            // Copy to host
            h_geom = new SphereGeom[num_hitables];
            checkCudaErrors(cudaMemcpy(h_geom, geom, num_hitables*sizeof(SphereGeom), cudaMemcpyDeviceToHost));
        } else {
            // UM modes: use unified memory
            checkCudaErrors(cudaMallocManaged((void **)&geom, num_hitables*sizeof(SphereGeom)));
            if (mem_mode == MEM_UM_PREFETCH) {
                checkCudaErrors(cudaMemPrefetchAsync(geom, num_hitables*sizeof(SphereGeom), 0));
            } else if (mem_mode == MEM_UM_ADVISE) {
                checkCudaErrors(cudaMemAdvise(geom, num_hitables*sizeof(SphereGeom), cudaMemAdviseSetPreferredLocation, 0));
                checkCudaErrors(cudaMemAdvise(geom, num_hitables*sizeof(SphereGeom), cudaMemAdviseSetAccessedBy, cudaCpuDeviceId));
            }
            export_sphere_data<<<1,1>>>(d_list, geom, num_hitables);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            
            // Prefetch to CPU if using prefetch mode
            if (mem_mode == MEM_UM_PREFETCH) {
                checkCudaErrors(cudaMemPrefetchAsync(geom, num_hitables*sizeof(SphereGeom), cudaCpuDeviceId));
                checkCudaErrors(cudaDeviceSynchronize());
            }
            h_geom = geom;  // Can access directly in UM modes
        }
        
        // Build BVH on CPU
        BVHNode *h_bvh = build_bvh_cpu(h_geom, num_hitables, num_bvh_nodes);
        
        // Allocate and transfer BVH based on memory mode
        if (mem_mode == MEM_EXPLICIT) {
            checkCudaErrors(cudaMalloc((void **)&d_bvh, num_bvh_nodes*sizeof(BVHNode)));
            checkCudaErrors(cudaMemcpy(d_bvh, h_bvh, num_bvh_nodes*sizeof(BVHNode), cudaMemcpyHostToDevice));
        } else {
            checkCudaErrors(cudaMallocManaged((void **)&d_bvh, num_bvh_nodes*sizeof(BVHNode)));
            memcpy(d_bvh, h_bvh, num_bvh_nodes*sizeof(BVHNode));
            
            if (mem_mode == MEM_UM_PREFETCH) {
                checkCudaErrors(cudaMemPrefetchAsync(d_bvh, num_bvh_nodes*sizeof(BVHNode), 0));
            } else if (mem_mode == MEM_UM_ADVISE) {
                checkCudaErrors(cudaMemAdvise(d_bvh, num_bvh_nodes*sizeof(BVHNode), cudaMemAdviseSetPreferredLocation, 0));
                checkCudaErrors(cudaMemAdvise(d_bvh, num_bvh_nodes*sizeof(BVHNode), cudaMemAdviseSetReadMostly, 0));
            }
        }
        
        clock_t bvh_stop = clock();
        double bvh_time = ((double)(bvh_stop - bvh_start)) / CLOCKS_PER_SEC * 1000.0;
        std::cerr << "BVH built with " << num_bvh_nodes << " nodes, took " << bvh_time << " ms.\n";
        
        // Cleanup temporary data
        checkCudaErrors(cudaFree(geom));
        if (mem_mode == MEM_EXPLICIT) {
            delete[] h_geom;
        }
        delete[] h_bvh;
    }

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, d_world, d_rand_state, use_bvh, d_bvh, d_list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Handle data transfer based on memory mode
    if (mem_mode == MEM_EXPLICIT) {
        checkCudaErrors(cudaMemcpy(h_fb, fb, fb_size, cudaMemcpyDeviceToHost));
    } else if (mem_mode == MEM_UM_PREFETCH) {
        checkCudaErrors(cudaMemPrefetchAsync(fb, fb_size, cudaCpuDeviceId));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    // For MEM_UM and MEM_UM_ADVISE, no explicit action needed - automatic migration
    
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    clock_t ppm_start = clock();
    std::ofstream outfile("out.ppm");
    outfile << "P3\n" << nx << " " << ny << "\n255\n";
    
    vec3 *output_fb = (mem_mode == MEM_EXPLICIT) ? h_fb : fb;
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int ir = int(255.99*output_fb[pixel_index].r());
            int ig = int(255.99*output_fb[pixel_index].g());
            int ib = int(255.99*output_fb[pixel_index].b());
            outfile << ir << " " << ig << " " << ib << "\n";
        }
    }
    outfile.close();
    clock_t ppm_stop = clock();
    double ppm_time = ((double)(ppm_stop - ppm_start)) / CLOCKS_PER_SEC * 1000.0;
    std::cerr << "Image saved to out.ppm, took "<< ppm_time <<" ms.\n";

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1,1>>>(d_list,d_world,d_camera,num_hitables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));
    
    // Free host buffer if using explicit mode
    if (h_fb) {
        delete[] h_fb;
    }
    
    // Free BVH if allocated
    if (d_bvh) {
        checkCudaErrors(cudaFree(d_bvh));
    }

    cudaDeviceReset();
}
