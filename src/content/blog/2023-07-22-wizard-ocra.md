---
title: WizardLM(Coder) 和 Ocra 的一些理解
pubDatetime: 2023-07-22
description: 介绍一下最近看到的两篇关于 SIFT 数据相关的非常好的论文 WizardLM(WizardCoder) 和 Ocra，以及我对这个问题的一些思考
tags:
  - SIFT
  - LLM
  - CoT
  - Evolution
draft: true
---

> LLM 一般分为两个阶段的训练，第一个阶段是 unsupervised pre-train, 不仅需要大量的 data，也需要很多计算资源；第二个阶段是 alignment，不仅包括 instruction tuning，还有 RLHF，这个阶段需要的计算资源和数量都远小于 pre-train，所以可以看到几乎全世界所有的机构都在 alignment 流程上分布式想 idea，也做了很多开源的工作。
>
> 最开始大家都只能在一些比较差的开源 base model 上做一些工作，做的效果也都差强人意。
> 但是随着 LLaMA[^1] 的开源，这个问题就不复存在了，同时由于开源社区可以尽情调用 gpt3.5 和 gpt4 来生成 instruction data，alignment 阶段做的工作也有了非常 promising 的进展。
> 其中最有代表性的就是 WizardLM 和 Ocra，下面介绍一下这两篇工作以及我对 alignment 上的一些思考。

## Table of contents

## WizardLM

首先介绍 WizardLM[^2] 这篇论文，这篇论文主要 focus 在 instruction data 的自动生成上。
因为学界没有那么多商业上的限制，所以可以通过 gpt3.5/gpt4 来生成 instruction data，这样就不用依赖复杂的人工标注也可以获得大量的 instruction data，
代表性的工作如 self-instruct[^3], [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) 等等。

不过这些工作都存在一些共同的问题，生成的 instruction data 不仅在多样性上有欠缺，同时在复杂度上也不够，这导致 instruction 和人类标注相比存在差距，在 LLM alignment 阶段获得效果也就非常一般了。

## WizardCoder

WizardCoder[^4]

## Ocra

## Reference

[^1]: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
[^2]: [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
[^3]: [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
[^4]: [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)

wizardLM and orca

While these models can produce content that matches the style of their teachers, they often fall short in terms of the reasoning and comprehension skills displayed by the larger foundation models.

“model imitation is a false promise” since “broadly matching ChatGPT using purely imitation would require (1) a concerted effort to collect enormous imitation datasets and (2) far more diverse and higher quality imitation data than is currently available.

Simple instructions with limited diversity
Self-Instruct paper, exhibit limitations in diversity and complexity

Task diversity and data scaling.

Explanation tuning

limited ability to trace the reasoning process of the LFM

Why sft can increase the reasoning performance of LLM?
pretrain 注入知识，sft 学习 format
pretrain 阶段学习的知识并不能 unleash
