---
title: "WizardCoder 和 Ocra 的一些理解"
pubDatetime: 2023-07-22
description: 介绍一下最近看到的两篇关于 SIFT 数据相关的非常好的论文 WizardLM(WizardCoder) 和 Ocra，以及我对这个问题的一些思考
tags:
  - SIFT
  - LLM
  - CoT
  - Evolution
---

LLM 一般分为两个阶段的训练，第一个阶段是 unsupervised pre-train, 不仅需要大量的 data，也需要很多计算资源；第二个阶段是 alignment，不仅包括 instruction tuning，还有 RLHF，这个阶段需要的计算资源和数量远小于 pre-train，所以可以看到几乎全世界所有的机构都在 alignment 阶段进行分布式想 idea。

本来大家还在等米下锅,随着 LLaMA[^1] 的开源，

## Reference

[^1]: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)

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
需要好好想想？
