---
title: Parallel Reduction Optimization with CUDA
pubDatetime: 2023-08-27
description: test
tags:
  - CUDA
  - Reduction
  - Parallel Computing
  - GPU
  - Engineering
comments: true
draft: true
---

> Reduction is a classic problem in parallel computing. Given an array, the task is to compute its `sum`, `min`, `max`, or `mean`. This operation serves as a fundamental data-parallel primitive.
>
> While the naive approach involves using a for-loop to iterate through each element and compute the result, this method is not the most efficient. So, how can CUDA be employed to optimize this process?
>
> In this article, we will walk you through various optimization techniques to enhance your reduction code step by step.

## Table of Contents

## Introduction

Suppose you want to calculate the sum of an array containing `N` integers. While a simple for-loop could iterate through each element to compute the sum, this approach becomes inefficient for large arrays.

```c
int cpuSum(int* data, size_t size) {
  int sum = 0;
  for (int i = 0; i < size; ++i) {
    sum += data[i];
  }
  return sum;
```

To speed up the process, parallel computation is essential. Fortunately, addition is both associative and commutative, allowing us to sum the array elements in any order. Moreover, addition is a low-arithmetic-intensity operation, so optimizing for peak bandwidth is crucial.

One might think that synchronizing across all thread blocks would make it easy to reduce very large arrays.
In this scenario, a global sync could occur after each block produces its partial result. Once all blocks reach this sync point, the process could continue **recursively**, thereby significantly reducing the scale of the arrays involved.

However, global synchronization is not feasible in CUDA. Implementing such a feature in hardware would be costly, especially for GPUs with a high processor count. Additionally, requiring block synchronization would compel programmers to run fewer blocks to avoid deadlock, potentially reducing overall efficiency.

For **deadlock**, imagine a situation where some blocks start executing and reach the sync point, waiting for the other blocks to catch up. While others cannot start executing because GPU's resources are full occupied by the blocks are waiting for sync.

A common workaround to avoid the need for global synchronization is to use kernel decomposition. Kernel launches can effectively serve as global synchronization points, offering the advantage of negligible hardware overhead and low software overhead.

In this approach, the code for all levels remains consistent, and the kernel is invoked recursively. The input arrays are divided into smaller chunks, with each block responsible for calculating the partial sum of its respective chunk.
An example of kernel decomposition is shown below.

<img src="/public/assets/cuda-reduce/kernel_decompose.png" width=600>

Next, let's walk you step by step through the optimization process.

## Baseline

First, let's examine the baseline version which is implemented with CPU, as depicted in the figure below.

<img src="./cpu_recursive.png" width=500>

The code snippet that follows is a recursive implementation using the interleaved pair approach.

```c
int cpuReduceSum(int* data, size_t size) {
  // terminate check
  if (size == 1) return data[0];

  // renew the stride
  size_t stride = size / 2;

  // in-place reduction
  for (int i = 0; i < stride; ++i) { data[i] += data[i + stride]; }
  return cpuReduceSum(data, stride);
}
```

To profile the baseline performance, we conducted a reduction on an array with 16M(2^24) elements. The results are as follows:

```bash
./reducesum starting reduction at device 0: NVIDIA GeForce RTX 3090     with array size 16777216    grid 32768 block 512
cpu reduce elapsed 54.432869 ms, bandwidth 1.232874 GB/s, cpu_sum: 2139353471
```

## Reduction#1

Now, let's move on to implementing the first version of the reduction algorithm using CUDA. This technique is known as interleaved addressing, also commonly referred to as **pairing**. The illustration below provides a visual explanation of the concept.

<img src="interleaved_addressing.png" width=600>

In this approach, the array is divided into separate blocks. Threads within each block are then paired to carry out the reduction. Take note of the for-loop in the code snippet below: the stride doubles with each iteration.
Threads numbered (2 \* stride) will perform the addition operation at each step. Ultimately, the sum will be stored in the first element of the array.

```c
__global__ void reduceNeighbored(int* g_idata, int* g_odata, size_t n) {
  unsigned int tid = threadIdx.x;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  int* i_data = g_idata + blockDim.x * blockIdx.x;

  for (int stride = 1; stride < blockDim.x; stride <<= 1) {
    // 2*stride threads execute
    if ((tid % (2 * stride) == 0)) { i_data[tid] += i_data[tid + stride]; }
    // synchronized within block
    __syncthreads();
  }
  if (tid == 0) g_odata[blockIdx.x] = i_data[0];
}
```

The reduction#1 results are as follows:

```bash
cpu reduce elapsed 54.432869 ms, bandwidth 1.232874 GB/s, cpu_sum: 2139353471
reduction#1 elapsed 0.677109 ms, bandwidth 99.110909 GB/s, gpu_sum: 2139353471
```

See what we did there? We achieved a 80x speedup with the first version CUDA implementation!
But we can do better.

## Reduction#2

What's the problem of reduction#1?

The answer is **divergence**. In the for-loop, only half of the threads are active at each iteration. The other half is idle. This phenomenon is known as **warp divergence**.

For example, in the first iteration, only 16 threads ara active in a warp which contains 32 threads. In the second iteration, only 8 threads are active. There are some many threads are idle in a warp.

We can just modify the index of the for-loop to avoid divergence. As shown in the figure below, we use coalesced addressing to avoid divergence.

<img src="no_divergence.png" width=700>

## Reference

- [Professional CUDA C Programming](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
- [cuda webinar2](https://developer.download.nvidia.cn/assets/cuda/files/reduction.pdf)
- [避免分支分化](https://face2ai.com/CUDA-F-3-4-%E9%81%BF%E5%85%8D%E5%88%86%E6%94%AF%E5%88%86%E5%8C%96/)
- [循环展开](https://face2ai.com/CUDA-F-3-5-%E5%B1%95%E5%BC%80%E5%BE%AA%E7%8E%AF/)
