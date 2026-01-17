#ifndef BVHH
#define BVHH

#include "vec3.h"
#include "ray.h"
#include "hitable.h"
#include <algorithm>

// Sphere geometry data (for BVH construction)
struct SphereGeom {
    vec3 center;
    float radius;
    int list_idx;  // Index in d_list array
    
    // Compute bounding box
    void get_bbox(vec3& min_out, vec3& max_out) const {
        min_out = center - vec3(radius, radius, radius);
        max_out = center + vec3(radius, radius, radius);
    }
    
    vec3 centroid() const {
        return center;
    }
};

// BVH node
struct BVHNode {
    vec3 bbox_min;
    vec3 bbox_max;
    int left_child;   // Index of left child, or -1 if leaf
    int right_child;  // Index of right child, or -1 if leaf
    int sphere_idx;   // Valid only for leaf nodes (list_idx)
    
    __device__ __host__ bool is_leaf() const {
        return left_child == -1;
    }
};

// AABB hit test
__device__ inline bool aabb_hit(const vec3& bbox_min, const vec3& bbox_max,
                                const ray& r, float t_min, float t_max) {
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (bbox_min[a] - r.origin()[a]) * invD;
        float t1 = (bbox_max[a] - r.origin()[a]) * invD;
        if (invD < 0.0f) {
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }
    return true;
}

// BVH traversal on GPU
__device__ bool bvh_hit(const BVHNode* bvh_nodes, hitable** d_list,
                        const ray& r, float t_min, float t_max, hit_record& rec) {
    if (bvh_nodes == nullptr) return false;
    
    // Stack-based traversal (no recursion)
    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;  // Start from root
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    hit_record temp_rec;
    
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        const BVHNode& node = bvh_nodes[node_idx];
        
        // Test AABB
        if (!aabb_hit(node.bbox_min, node.bbox_max, r, t_min, closest_so_far)) {
            continue;
        }
        
        if (node.is_leaf()) {
            // Leaf node - test sphere
            if (d_list[node.sphere_idx]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        } else {
            // Internal node - push children
            if (node.right_child != -1) {
                stack[stack_ptr++] = node.right_child;
            }
            if (node.left_child != -1) {
                stack[stack_ptr++] = node.left_child;
            }
        }
    }
    
    return hit_anything;
}

//------------------------------------------------------------------------------
// CPU BVH construction functions
//------------------------------------------------------------------------------

// Compute bounding box for a range of spheres
inline void compute_bbox(const SphereGeom* spheres, int start, int end,
                        vec3& bbox_min, vec3& bbox_max) {
    bbox_min = vec3(FLT_MAX, FLT_MAX, FLT_MAX);
    bbox_max = vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (int i = start; i < end; i++) {
        vec3 smin, smax;
        spheres[i].get_bbox(smin, smax);
        
        for (int a = 0; a < 3; a++) {
            bbox_min.e[a] = fmin(bbox_min[a], smin[a]);
            bbox_max.e[a] = fmax(bbox_max[a], smax[a]);
        }
    }
}

// Build BVH recursively on CPU
inline int build_bvh_recursive(SphereGeom* spheres, int start, int end,
                               BVHNode* nodes, int& node_count) {
    int node_idx = node_count++;
    BVHNode& node = nodes[node_idx];
    
    // Compute bounding box for this node
    compute_bbox(spheres, start, end, node.bbox_min, node.bbox_max);
    
    int count = end - start;
    
    // Leaf node
    if (count == 1) {
        node.left_child = -1;
        node.right_child = -1;
        node.sphere_idx = spheres[start].list_idx;
        return node_idx;
    }
    
    // Find longest axis
    vec3 extent = node.bbox_max - node.bbox_min;
    int axis = 0;
    if (extent.y() > extent.x()) axis = 1;
    if (extent.z() > extent[axis]) axis = 2;
    
    // Sort along axis
    std::sort(spheres + start, spheres + end,
        [axis](const SphereGeom& a, const SphereGeom& b) {
            return a.centroid()[axis] < b.centroid()[axis];
        });
    
    // Split in the middle
    int mid = start + count / 2;
    
    // Build children
    node.sphere_idx = -1;
    node.left_child = build_bvh_recursive(spheres, start, mid, nodes, node_count);
    node.right_child = build_bvh_recursive(spheres, mid, end, nodes, node_count);
    
    return node_idx;
}

// Main BVH construction function (called from CPU)
inline BVHNode* build_bvh_cpu(SphereGeom* spheres, int num_spheres, int& num_nodes) {
    // Allocate enough space (worst case: 2*n-1 nodes for n leaves)
    int max_nodes = 2 * num_spheres;
    BVHNode* nodes = new BVHNode[max_nodes];
    
    int node_count = 0;
    build_bvh_recursive(spheres, 0, num_spheres, nodes, node_count);
    
    num_nodes = node_count;
    return nodes;
}

#endif
