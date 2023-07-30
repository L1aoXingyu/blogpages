---
title: 如何做 continued pre-train
pubDatetime: 2023-07-04
description: 介绍一下 continued pre-train
tags:
  - deep learning
  - LLM
  - pre-train
comments: true
---

在 LLM 中，为了获得一个好的 basemodel，往往都会使用 pre-train 的方式。但是通常 pre-train 都需要消耗很多资源，不管是算力还是数据。其实除了 pre-train 之外，还有一种方案叫做 continued pre-train，这篇文章会简要介绍一下这种方案。

## 0x0 什么是 continued pre-train

continued pre-train 从名字就可以看出，他是在 pre-train 的基础上继续训练，是一个 secondary stage 任务。他的 loss function 和 pre-train 完全一致，在 decoder-only 的架构下是以 AR[^1] 做为目标进行训练。

我们知道 SIFT[^2] 也是一个 pre-train 的 secondary stage 任务，那么他们之间有什么区别呢？
他们之间的区别主要有下面这几个方面：1）SIFT 的数据规模和 continued pre-train 不同，远远少于 pre-train 阶段，而且 SIFT 如果使用过多的数据容易出现 overfitting 的问题；2）SIFT 的数据对质量和多样性的要求比较高；3）SIFT 和 pre-train 的 loss objective 有轻微区别，SIFT 是部分的 AR，会 mask 掉 instruction 部分，只对 response 部分算 loss。

## 0x1 为什么要做 continued pre-train

continued pre-train 介于 pre-train 和 SIFT 之间，他的存在有着他的独特价值。

目前存在的基本共识是 pre-train 阶段将知识注入模型，在 SIFT 阶段学习 style 和 instruction follow 的能力。所以当我们需要注入新知识的时候，continued pre-train 就派上用场了。

当注入的新知识只是很小的一部分，pre-train from scratch 并不是一个高效的选择，另外有的时候我们只能拿到模型训练好的权重，并不能拿到 pre-train data，所以想要重新 pre-train 也是不可能的。

有的时候，针对特定 domain 进行 continued pre-train 能让模型的能力更 focus 在对应的 domain 上获得更好的能力。

## 0x2 continued pre-train 有什么挑战

虽然说 continued pre-train 和 pre-train 非常类似，就是在 pre-train 模型的基础上继续训练，但是还是会有一些新的问题。

### 0x2.1 词表扩充问题

我们知道模型在进行训练之后，需要对 corpus 做 tokenize，如果要做 continued pre-train 的 corpus 和之前 pre-train 的不一致，就会遇到 tokenize 效率的问题。

比如之前 Meta 开源的 LLaMa 模型，它有着非常强的性能，但是其在英文语料上进行训练的，如果希望他支持中文，就需要用中文进行 continued pre-train。但是 LLaMa 的 tokenizer 是在英文语料上训练的[^3]，在遇到没有见过的中文时候，会怎么样呢？

实际上，现在的 tokenizer 基本都是基于 BPE[^4] 设计的，所以原理上能够支持所有的字符，当他遇到没有见过的字符时，它会 fall back to bytes，比如 "我" 这个字符就可以用 `E6 88 91` 来表示，最终也可以被 tokenizer 编码。

但是这种策略会引入效率的问题，不仅会影响 tokenizer 的 encode/decode 效率，还会显著增加编码之后的序列长度，这也直接影响了模型的训练效率，相当相同的 token 数包含的信息更少了。

这篇文章[EFFICIENT AND EFFECTIVE TEXT ENCODING FOR CHINESE LLAMA AND ALPACA](https://arxiv.org/pdf/2304.08177.pdf) 提出可以通过扩充词表来解决这个问题。方法非常简单，具体来说就是对新 domain corpus 用 SentencePiece 重新训练一个 tokenizer，然后将新的词表和之前旧的词表重新合并在一起，在合并的时候取他们的并集，同时扩充 embedding，保留之前训好的 embedding，在后面初始化新词表的 embedding，这样可以保证之前训练好的 embedding 不受影响。

因为新加的词表是随机初始化的，和之前训好的词表在分布式上是不一致的，所以可以考虑固定网络的其它层，先少规模 tuning embedding 的部分。

### 0x2.2 防止遗忘的问题

当我们对模型注入新知识的时候，可能导致模型遗忘之前学习到的知识，比如用 code 对模型进行 continued pre-train 时，一些 text-based tasks 性能会降低。为了缓解这个问题，可以在 data mixture 的时候，采样之前 domain 的数据，和当前 domain 的数据按照某种比例进行混合。除此之外，还可以考虑使用 EMA 或者是 mean-teacher 的方案来更新模型，防止模型过快地遗忘之前的知识。

### 0x2.3 如何高效 tuning

continued pre-train 的一个优势是不需要从头训练之前见过的 token，只需要训练新增的 token。为了更进一步降低成本，还可以在 continued pre-train 阶段使用 LoRA[^5] tuning，这种方案可以不用训练模型的所有参数，只需要一部分很小的参数，实现更小的显存和更快的训练速度。

## 0x3 总结

这篇文章主要介绍一下 continued pre-train 的好处以及其应用场景，分析了 continued pre-train 面临的一些问题。笔者水平有限，如有遗漏或者问题，欢迎指出。

[^1]: Auto Regressive.
[^2]: Supervised Instruction Fine-Tuning.
[^3]: Less than one thousand Chinese character.
[^4]: Byte-Pair Encoding.
[^5]: https://github.com/microsoft/LoRA
