---
title: Benchmark for LLM Inference
pubDatetime: 2023-08-20
description: Introduce some metrics for LLM inference benchmarking
tags:
  - LLM
  - Benchmark
  - Profile
  - Inference
  - Transformer
  - Accelerate
comments: true
featured: true
draft: true
---

> As the development of Large Language Models (LLMs) progresses, the cost of inference becomes increasingly significant. Unlike training, which is typically performed once, inference is executed repeatedly. Consequently, the cost of inference begins to dominate the total cost of utilizing a model.
>
> Given the importance of inference cost, how can we effectively benchmark it? > In this post, we will explore various metrics designed specifically for benchmarking the inference process in LLMs.

## Table of contents

## LLM Inference

Inference is the process of using a trained model to make predictions on new data. In the context of Large Language Models (LLMs), inference refers to generating text based on a given prompt. For example, given the prompt "The quick brown fox jumps over the lazy dog," an LLM might generate a continuation of this text.

Unlike traditional Deep Learning (DL) models, LLMs are autoregressive, meaning they generate text one token at a time. This characteristic distinguishes them from other models and influences how they are utilized.

There are typically two types of LLM serving scenarios: **offline** and **online**.
In the **offline** scenario, inference requests are usually batched, and latency is not a significant concern.
Conversely, in the **online** scenario, inference requests are processed as a continuous stream, making latency a critical factor to consider.

## Metrics

As described above, **throughout** and **latency** are two core metrics for LLMs inference benchmarking. Let's drill down into each of them.

### Throughput

[Throughput](https://en.wikipedia.org/wiki/Network_throughput) is a measure of how many units of information a system can process within a given time frame. In the context of Large Language Models (LLMs), throughput can be understood as the number of users that can be served simultaneously.

In real-world applications, each request (or prompt) often varies in context length, necessitating the simulation of these variations and the design of a suitable method to test throughput. Another critical factor to consider is the length of the generated text. In practical scenarios, this text can also vary in length, and accommodating these variations is essential for an accurate assessment of system performance.

Considering these two factors, we can create a dumpy dataset which contains a list of prompts with different lengths.

### Latency

[Latency](<https://en.wikipedia.org/wiki/Latency_(engineering)>) is the time it takes for a system to respond to a request.

## Conclusion

## Reference
