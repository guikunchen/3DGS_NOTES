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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
    template <typename T>
    static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
    {
        std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); // 对齐后的偏移量
        ptr = reinterpret_cast<T*>(offset);           // 转换为指针，并通过引用返回
        chunk = reinterpret_cast<char*>(ptr + count); // 更新 chunk 指针，指向分配的内存块之后的位置
    }

    struct GeometryState
    {
        size_t scan_size;         // 计算前缀和（扫描）时的 temp_storage_bytes，即辅助空间的大小
        float* depths;            // 对所有 Gaussions，在投影坐标系下的深度
        char* scanning_space;     // 计算前缀和（扫描）时的 d_temp_storage，即辅助空间的起始指针
        bool* clamped;            // 对所有 Gaussions，预处理从 SH 算 RGB 的时候被裁剪到正值，keep track of this for the backward pass
        int* internal_radii;      // 对所有 Gaussions，图像坐标系下估算为圆的半径 my_radius
        float2* means2D;          // 图像坐标系下所有 Gaussions 的均值
        float* cov3D;             // 对所有 Gaussions，世界坐标系下的协方差（6 个一组）
        float4* conic_opacity;    // 对所有 Gaussions，图像坐标系下 2D 协方差矩阵的逆和不透明度 (3 + 1 = 4)
        float* rgb;               // 对所有 Gaussions，预处理从 SH 算 RGB 的结果
        uint32_t* point_offsets;  // 每个 Gaussions 触碰 tiles 的个数的前缀和
        uint32_t* tiles_touched;  // 每个 Gaussions 触碰 tiles 的个数
        // 4 + 8 * 10 < 128，这些指针所占大小不超过一个小块(128)。以下类推，所以计算 required 把 fromChunk 返回值加上 128 就够了
        static GeometryState fromChunk(char*& chunk, size_t P);
    };

    struct ImageState
    {
        uint2* ranges;        // 每个 tile 需要跟 BinningState.point_list 里的哪些 Gaussions 做运算，长为 num_rendered * 2
        uint32_t* n_contrib;  // 长为 pixel 的总数，其中每个值是 ranges.x ~ ranges.y 中有贡献的数量（可能因为 不透明度累计 < 0.0001f 而提前终止）
        float* accum_alpha;   // 长为 pixel 的总数，其中每个值是终止时的不透明度，这些值在 backward 时有用

        static ImageState fromChunk(char*& chunk, size_t N);
    };

    struct BinningState
    {
        size_t sorting_size;                // SortPairs 时的 temp_storage_bytes，即辅助空间大小
        uint64_t* point_list_keys_unsorted; // |tile id, Gaussion depth| 的未排序 list
        uint64_t* point_list_keys;          // |tile id, Gaussion depth| 的排序后 list
        uint32_t* point_list_unsorted;      // Gaussion id 的未排序 list
        uint32_t* point_list;               // Gaussion id 的排序 list
                                            // 注意每个 list 长为 num_rendered，里面的东西可能是有重复的，其含义参见 rasterizer_impl.cu 286 ~ 306
        char* list_sorting_space;           // SortPairs 时的 d_temp_storage，即辅助空间的起始指针

        static BinningState fromChunk(char*& chunk, size_t P);
    };

    template<typename T> 
    size_t required(size_t P)
    {
        char* size = nullptr;
        T::fromChunk(size, P);
        return ((size_t)size) + 128;
    }
};