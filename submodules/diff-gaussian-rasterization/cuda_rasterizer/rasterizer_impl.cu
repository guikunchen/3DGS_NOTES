/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB on the CPU.
// Most Significant Bit，二进制表示中最左边的值
uint32_t getHigherMsb(uint32_t n)
{
    uint32_t msb = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
        msb++;
    return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool* present)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p_view;
    present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
    int P,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* gaussian_keys_unsorted,
    uint32_t* gaussian_values_unsorted,
    int* radii,
    dim3 grid)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    // Generate no key/value pair for invisible Gaussians
    if (radii[idx] > 0)
    {
        // Find this Gaussian's offset in buffer for writing keys/values.
        uint32_t off = (idx == 0) ? 0 : offsets[idx - 1]; // 利用之前算的前缀和来确定目标 chunk 地址
        uint2 rect_min, rect_max;

        getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a 
        // key/value pair. The key is |  tile ID  |      depth      |,
        // and the value is the ID of the Gaussian. Sorting the values 
        // with this key yields Gaussian IDs in a list, such that they
        // are first sorted by tile and then by depth. 
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                gaussian_keys_unsorted[off] = key;
                gaussian_values_unsorted[off] = idx;
                off++; // 下一个要复制过去的 chunk
            }
        }
    }
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= L)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = point_list_keys[idx]; // 有序的 key（前 32 位）：1, 1, ..., 2, ..., ...
    uint32_t currtile = key >> 32;
    // 每个 thread 设置 current tile 的 min(x) 和 prevent tile 的 max(y)，各司其职没有冲突
    if (idx == 0)
        ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = point_list_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            ranges[prevtile].y = idx;
            ranges[currtile].x = idx;
        }
    }
    if (idx == L - 1)
        ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
    int P,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    bool* present)
{
    checkFrustum << <(P + 255) / 256, 256 >> > (
        P,
        means3D,
        viewmatrix, projmatrix,
        present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{ // fromChunk 用于申请空间地址，可以参考下面 BinningState 的解读
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P); // 同样是分两次调用，参考下面 SortPairs 的注释
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{ // fromChunk 用于申请空间地址，可以参考下面 BinningState 的解读
    ImageState img;
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);               // 分配 uint32_t * P
    obtain(chunk, binning.point_list_unsorted, P, 128);      // 再分配 uint32_t * P
    obtain(chunk, binning.point_list_keys, P, 128);          // 再分配 uint64_t * P
    obtain(chunk, binning.point_list_keys_unsorted, P, 128); // 再分配 uint64_t * P
    // SortPairs' documentation: When d_temp_storage is nullptr, no work is done and the required allocation size is returned in temp_storage_bytes
    // 因为这个函数需要一定的空间来排序，所以预先给它申请空间。所以这个函数是这样设计的：两阶段调用，第一次先得到需求尺寸，据此为其申请空间，后面再真正调用计算
    cub::DeviceRadixSort::SortPairs( 
        nullptr, binning.sorting_size,
        binning.point_list_keys_unsorted, binning.point_list_keys,
        binning.point_list_unsorted, binning.point_list, P);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128); // 再分配 1(char) * sorting_size 那么大的空间
    return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(  // 可以把这个当成 main 函数
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const int P, int D, int M, // points number, SH degree(0 -> 3), SH 的维度(16)
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const float tan_fovx, float tan_fovy,
    const bool prefiltered,
    float* out_color,
    int* radii,
    bool debug)
{
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Dynamically resize image-based auxiliary buffers during training
    size_t img_chunk_size = required<ImageState>(width * height);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");

    // FORWARD 空间里面的基本都是 kernel function（那些带 < <<xxx, yyy> > 的调用）了，CUDA 会把资源分成 block 级别(tile) 和 thread 级别(each pixel in a tile)，每个 thread 全部并行地去做下面的代码，通过一些标识函数找到自己的位置。粗浅的引入可参考 https://zhuanlan.zhihu.com/p/129375374
    // Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
    // preprocess 把每个高斯投影到图像上
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered
    ), debug)

    // ---开始--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表
    // Compute prefix sum over full list of touched tile counts by Gaussians
    // E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8] 计算前缀和
    // in: tiles_touched, out: points_offset，然后取出来存到 num_rendered 里
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

    // Retrieve total number of Gaussian instances to launch and resize aux buffers
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

    // chunk 的意思是一块大内存，这里是为 BinningState 申请空间，其中形状如下
    // | unsigned long |       uint64_t * P       |  uint64_t * P  |    uint32_t * P    | uint32_t * P |       char       |
    // | sorting_size | point_list_keys_unsorted | point_list_keys | point_list_unsorted | point_list | list_sorting_space |
    // point_list_keys-point_list 中的值分别为 key-value，每个跟一个 tile 与一个 Gaussion 对应
    // 也就是一共要算 num_rendered 次 tile-and-Gaussion，其中每个 tile 内的每个像素由一个 thread 执行 (pixel-and-Gaussion)
    // 去看后面 duplicateWithKeys() 内部，每个 index 对应一个 key-value 键值对，key 是 |tile ID|depth|，value 是 Gaussion idx
    // key 既存 id 又存 Gaussion 深度，如果单从符合直觉的角度，深度放在 value 更好理解一些
    // 但这里这样做的考虑是，后面 SortPairs 的时候，能同时排序 tiles 以及它对应的 Gaussions 的深度。一次排两个东西的序，妙绝！
    size_t binning_chunk_size = required<BinningState>(num_rendered);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

    // For each instance to be rendered, produce adequate [ tile | depth ] key 
    // and corresponding dublicated Gaussian indices to be sorted
    // 排序前，顺序以 Gaussion id 为序
    // e.g., index 0 for tile x with Gaussion 0, index 1 for tile x with Gaussion 0;
    //       index 2 for tile a with Gaussion 1, index 3 for tile b with Gaussion 1, index 4 for tile c with Gaussion 1
    // 排序后，就是以 tile 为主序，深度为次序了
    // e.g., index 0 for tile 0 with Gaussion x, index 1 for tile 0 with Gaussion y, ... 
    //       (assume there are 10 Gaussions touched tile 0, and these Gaussions are sorted by depth)
    //       index 10 for tile 1 with Gaussion t, ...
    duplicateWithKeys << <(P + 255) / 256, 256 >> > (  // 根据 tile 复制 Gaussian （对每个 Gaussion，把它复制到 tiles 里）
        P,
        geomState.means2D,
        geomState.depths,
        geomState.point_offsets,
        binningState.point_list_keys_unsorted,
        binningState.point_list_unsorted,
        radii,
        tile_grid)
    CHECK_CUDA(, debug)

    int bit = getHigherMsb(tile_grid.x * tile_grid.y); // 算出 tile id 部分最高位，后面排序时只用考虑 0 ~ 32 + bit

    // Sort complete list of (duplicated) Gaussian indices by keys
    CHECK_CUDA(cub::DeviceRadixSort::SortPairs( // 在每个 tile 中，对复制后的所有 Gaussians 进行排序，排序的结果可供平行化渲染使用
        binningState.list_sorting_space,
        binningState.sorting_size,
        binningState.point_list_keys_unsorted, binningState.point_list_keys,
        binningState.point_list_unsorted, binningState.point_list,
        num_rendered, 0, 32 + bit), debug)

    CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

    // Identify start and end of per-tile workloads in sorted list
    if (num_rendered > 0) {
        // 根据有序的 Gaussian 列表，判断每个 tile 需要跟哪一个 range 内的 Gaussians 进行计算
        identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
            num_rendered,
            binningState.point_list_keys,
            imgState.ranges);
    }
    CHECK_CUDA(, debug)
    // ---结束--- 通过视图变换 W 计算出像素与所有重叠高斯的距离，即这些高斯的深度，形成一个有序的高斯列表

    // Let each tile blend its range of Gaussians independently in parallel
    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(  // 核心渲染函数，具体实现在 forward.cu/renderCUDA
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        out_color), debug)

    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
    const int P, int D, int M, int R,
    const float* background,
    const int width, int height,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* scales,
    const float scale_modifier,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const float tan_fovx, float tan_fovy,
    const int* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dopacity,
    float* dL_dcolor,
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    bool debug)
{
    // 前面 forward 过程是用 CUDA 实现的，所以没法依靠 Pytorch 构建计算图来自动反向传播，因此需要手动设置反向过程，也就是反过来计算梯度
    // forward 过程看得细一点，反向过程就不细注释了
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Compute loss gradients w.r.t. 2D mean position, conic matrix,
    // opacity and RGB of Gaussians from per-pixel loss gradients.
    // If we were given precomputed colors and not SHs, use them.
    const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    CHECK_CUDA(BACKWARD::render(
        tile_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        background,
        geomState.means2D,
        geomState.conic_opacity,
        color_ptr,
        imgState.accum_alpha,
        imgState.n_contrib,
        dL_dpix,
        (float3*)dL_dmean2D,
        (float4*)dL_dconic,
        dL_dopacity,
        dL_dcolor), debug)

    // Take care of the rest of preprocessing. Was the precomputed covariance
    // given to us or a scales/rot pair? If precomputed, pass that. If not,
    // use the one we computed ourselves.
    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    CHECK_CUDA(BACKWARD::preprocess(P, D, M,
        (float3*)means3D,
        radii,
        shs,
        geomState.clamped,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        cov3D_ptr,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        (glm::vec3*)campos,
        (float3*)dL_dmean2D,
        dL_dconic,
        (glm::vec3*)dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        (glm::vec3*)dL_dscale,
        (glm::vec4*)dL_drot), debug)
}