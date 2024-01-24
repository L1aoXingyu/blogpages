---
title: 8-bit KV Cache
pubDatetime: 2024-01-24
description: This blog introduces KV Cache quantization in LLM inference.
tags:
  - LLM
  - KV Cache
  - Quantization
  - Compression
  - Inference
  - Transformer
ogImage: /assets/int8_kvcache/kv_cache.png
comments: true
featured: false
---

> KV Cache plays a vital role in Large Language Model (LLM) inference, although it requires considerable memory storage.
>
> In this blog post, we'll explore how quantization of KV Cache can help in saving memory. Let's delve into this topic.

## Table of Contents

## Introduction

Large Language Models (LLMs) pose a significant challenge in terms of inference due to their inherent mechanism complexity. Various methods have been developed to optimize LLM inference, such as [PagedAttention](https://vllm.ai/), [Continuous batching](https://www.usenix.org/system/files/osdi22-yu.pdf).

Quantization represents another efficacious strategy, as highlighted in a preceding blog post titled [LLM Quantization Review](https://sherlock-dev.netlify.app/posts/llm-quantization-review/). Typically, quantization can be employed either on weights or activations. The discourse often centers around weight-only quantization methods, yet quantizing activations or the kv cache is also a pivotal aspect in enhancing LLM inference speed.

To elucidate this, consider the example of a 7B model inference where the kv cache consumes nearly 84% of the GPU DRAM, thereby limiting the batch size. Similarly, in the case of a 175B model, the kv cache accounts for approximately 50% of the GPU DRAM. This substantial resource allocation to kv cache storage suggests that by reducing its size, it's feasible to enlarge the batch size. Such a modification would in turn ameliorate the compute and memory access ratio, promising a more efficient inference process.

<div align=center>
<img src="/assets/int8_kvcache/kv_cache_store.png" width=450>
</div>

## KV Cache

Before delving into the quantization of KV Cache, it's pertinent to introduce KV Cache briefly.

**What is KV Cache?**

During the sampling stage of transformer inference, there's a prefill phase (which can occur in parallel) followed by the generation of additional tokens sequentially. Both during prefill and token generation, self-attention is executed, necessitating kv values for each item in the current sequence. These vectors are housed in a matrix referred to as the KV Cache or past cache, typically structured as `[batch, seq_len, 2, num_head, head_dim]`.

Below is a simplified depiction of transformer inference with KV Cache. In the pre-fill stage, all context tokens are processed in parallel, and subsequently stored as KV Cache. When the first token is sampled, self-attention can be performed utilizing the KV Cache, avoiding redundant computations.

<div align=center>
<img src="/assets/int8_kvcache/kv_cache.png" width=550>
</div>

Post token generation, the `k` and `v` vectors are concatenated along the `seq_len` dimension. Hence, the byte count for storing KV Cache can be formulated as:

$$
2 \cdot 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}}
$$

Here, the initial factor of $2$ accounts for the two vectors, `k` and `v`, and the second $2$ represents the byte count (assuming 16-bit formats). KV Cache storage occurs across each layer, with the values being $n_{\text{heads}} \cdot d_{\text{head}}$ heads. It's crucial to note that this is per sample, hence KV Cache storage linearly correlates with `batch_size`.

**The Necessity of Quantization**

With a firm understanding of GPU storage pertaining to KV Cache and weights (as intermediate memory is negligible), let's exemplify using the Llama-70B model.

Given the parameter count, doubling this gives the byte count (not considering weight quantization). Therefore, the weight size for a 70B model is:

$$
70e12 \cdot 2 = 140e12 \text{bytes} \approx 140\text{GB}
$$

This size exceeds the capacity of a single GPU, necessitating at least two GPUs to accommodate all the weights, leaving $2 \cdot 80\text{GB} - 140\text{GB} = 20\text{GB}$ for our KV Cache. Is this sufficient? Revisiting our KV Cache memory per token equation for a 70B model, we have:

$$
2 \cdot 2 \cdot n_{\text{layers}} \cdot n_{\text{heads}} \cdot d_{\text{head}} \\ = 4 \cdot 80 \cdot 8192 \\ = 2,621,440 \text{bytes} \approx 0.0026 \text{GB}
$$

Subsequently, $20\text{GB} / 0.0026 \approx 7692$ tokens can be accommodated in our KV Cache with this GPU configuration. Given a maximum of 2048 tokens per request, the batch size is capped at 3.

This is suboptimal as higher batch sizes are desirable for efficiency, but we are hampered by capacity constraints. In scenarios like this, the quantization of KV Cache emerges as a pivotal solution to alleviate memory demands and bolster batch size potential.

On the other hand, during the decoding stage where a token is generated each time, the primary constraint arises from memory bounds. Employing a quantized KV Cache can mitigate the I/O bandwidth requirements, thereby enhancing the overall inference performance.

## Method

KV cache quantization essentially entails activation quantization, a task rendered challenging due to the activation outlier problem. This issue has been discussed in a previous blog post, [SmoothQuant and AWQ](https://sherlock-dev.netlify.app/posts/smoothquant-and-awq/), where the outlier problem is illustrated.

<div align=center>
<img src=/assets/sq_awq/activation.png width=300>
</div>

The activation outlier problem can be addressed by "smoothing" the activation, which involves dividing by a "smoothing" factor and scaling the weights accordingly. In this blog post, we'll proceed with the assumption that the activation has been "smoothed" during the model quantization stage, and thus, can be safely quantized. With 8-bit quantization, two data types are considered: `FP8` and `INT8`.

### FP8 KV Cache

Let's commence with a discussion on `FP8` KV Cache quantization, which is relatively straightforward as it bypasses the complexities associated with grouping and scale-related issues.

Here's a depiction of the `FP8` format using the `e5m2` representation:

<div align=center>
<img src=/assets/int8_kvcache/fp8_dtype.png width=400>
</div>

Utilizing the `FP8` format allows for retaining the same exponential position as `FP16` without incurring the storage cost of scales. Concurrently, the de-quantization process is expedited as it merely requires padding zeros for the residual bits.

Below is a demonstration of how to implement `FP8` quantization using Triton:

```python
def f16_to_f8(x: torch.Tensor, dtypes=tl.float8e5) -> torch.Tensor:
    assert "cuda" in str(x.device), f"CUDA tensors only but got {x.device}"

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty_like(x, dtype=torch.int8)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']),)
    # manage cases where tensor is not contiguous, like ::2
    numel = x.untyped_storage().size() // x.element_size()
    kernel[grid](triton.reinterpret(ret, dtypes), x, numel, BLOCK_SIZE=512)
    return ret
```

Given that most of us lack access to H100, `torch.int8` can serve as the underlying data type for `FP8`. The conversion from `FP16` to `FP8` in Triton is straightforward, simply employ `triton.reinterpret(ret, tl.float8e5)` to view ret with `torch.int8` as the `FP8` format. Subsequently, utilize `tl.load(X+offs)` to load the original `FP16` data, then `tl.store(ret+offs)` to transition `FP16` to `FP8` format for storage.

The de-quantization procedure is akin; reinterpret the input as `tl.float8e5`, then apply `x.to(tl.float16)` to revert it back to `FP16` for subsequent computations.

Lastly, should you encounter an issue when setting `BLOCK_SIZE` to a smaller value like `128`:

```shell
Assertion `operands.size() % 4 == 0 && "FP8 casting only support tensors with 4-aligned sizes"' failed.
```

This can be remedied by adjusting `num_warps=1` since the default `num_wraps=4`. The underlying cause is the SIMD (Single Instruction, Multiple Data) programming model in Triton. When `BLOCK_SIZE=128` and `num_warps=4`, each thread processes merely one element as a warp encompasses `32` threads. Consequently, the operand size for each thread isn't `4-aligned`. Therefore, the number of warps can be determined by the formula `num_warps=BLOCK_SIZE // 32 // 4`.

### Int8 KV Cache

The Int8 KV Cache quantization process is somewhat similar to the FP8 process, but it includes an additional computation step. Imagine that the input activation is shaped like `[bs, seq_len, num_head, head_size]`. For INT8 quantization, we can handle it in different ways: per-tensor, channel-wise, or group-wise. Each method has its own trade-off in terms of accuracy and efficiency.

Let's focus on the group-wise method for clarity. In this approach, we reorganize the input shape into `[bs, seq_len, num_head, num_group, group_size]`, making sure that the product of `num_group` and `group_size` equals `head_size`. Within each group, the scale is determined by the maximum absolute value in the group, termed as `absmax(group)`.

Below is a basic example of Int8 KV Cache using the triton kernel:

```python
@triton.jit(debug=True)
def _fwd_kernel_int8_quantize_kv(
    K, out, out_scale,
    stride_k_b, stride_k_s, stride_k_h,
    stride_o_b, stride_o_s, stride_o_h,
    stride_os_b, stride_os_s, stride_os_h,
    num_groups,
    BLOCK_GROUP_NUM: tl.constexpr,
    BLOCK_GROUP_DIM: tl.constexpr
):
    cur_bs = tl.program_id(0)
    cur_index = tl.program_id(1)
    cur_head = tl.program_id(2)

    offs_g = tl.arange(0, BLOCK_GROUP_NUM)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM)

    # load src tensor
    offs_k = cur_bs * stride_k_b + cur_index * stride_k_s + cur_head * stride_k_h \
                + offs_g[:, None] * BLOCK_GROUP_DIM + offs_d[None, :]
    src_data = tl.load(K + offs_k, mask=offs_g[:, None] < num_groups, other=0.0)
    # quantize
    data_scale = (tl.max(tl.abs(src_data), axis=1) / 127.).to(tl.float16)
    q_src_data = tl.math.round(src_data / data_scale[:, None])
    q_src_data = tl.where(x > 127, 127, q_src_data)
    q_src_data = tl.where(x < -128, -128, q_src_data)
    q_src_data = q_src_data.to(tl.int8)

    # save quantized tensor and corresponding scales
    offs_o = cur_bs * stride_o_b + cur_index * stride_o_s + cur_head * stride_o_h \
                + offs_g[:, None] * BLOCK_GROUP_DIM + offs_d[None, :]
    offs_os = cur_bs * stride_os_b + cur_index * stride_os_s + cur_head * stride_os_h + offs_g
    tl.store(out + offs_o, q_src_data, mask=offs_g[:, None] < num_groups)
    tl.store(out_scale + offs_os, data_scale, mask=offs_g < num_groups)
```

This portion of the code is dedicated to quantizing the input tensors. The `scale` is calculated in real-time. Alternatively, Post-Training Quantization (PTQ) can be used to determine the `scale` beforehand and then input it into the kernel.

```py
data_scale = (tl.max(tl.abs(src_data), axis=1) / 127.).to(tl.float16)
q_src_data = tl.math.round(src_data / data_scale[:, None])
q_src_data = tl.where(x > 127, 127, q_src_data)
q_src_data = tl.where(x < -128, -128, q_src_data)
q_src_data = q_src_data.to(tl.int8)
```

Using a quantized KV cache significantly reduces memory usage. However, the dequantization process adds extra computational work in the kernel. This overhead can be lessened in situations where the context is large and data loading time of the KV cache will become longer.

## Summary

KV Cache is a crucial component in Large Language Model (LLM) inference, offering various optimizations for different aspects, like paged attention, which optimizes memory fragmentation.

This blog post focuses on the quantization aspect of KV Cache. By quantizing from fp16 to int8, we can halve the memory usage during runtime. While this process does introduce additional computational overhead, it remains significantly beneficial, especially in memory-constrained scenarios.

## Reference

- [lightllm](https://github.com/ModelTC/lightllm/tree/main)
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/)
- [vLLM](https://vllm.ai/)
- [LLM 推理性能优化探索](https://zhuanlan.zhihu.com/p/653735572)
