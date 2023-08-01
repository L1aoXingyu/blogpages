---
title: "如何在 OneFlow 中开发一个新的 UserOp"
pubDatetime: 2021-11-18
description: 记录在 oneflow 中开发 userOp 的流程以及中间遇到的一些问题
comments: true
tags:
  - deep learning
  - userOp
  - dl framework
---

## Table of contents

## Introduction

这篇文章主要记录了笔者学习使用 oneflow 开发 userOp 中的整个过程，将这个流程写成文章是为了进一步加深自己的学习和理解，毕竟输出才是最好的学习过程。

因为这篇文章是自己学习过程中的梳理，所以文章中可能会缺少一些背景知识的介绍，笔者会尽量弱化这些背景知识跟本文核心内容之间的联系，尽量用深度学习框架中都有的概念来讲解这些内容。

通过阅读这篇文章，你可以了解到 OneFlow 开发 UserOp 的整体流程以及流程中一些关键步骤的作用。

## 整体开发流程

要实现一个 Op 分为两个大的部分，分别是 Op 的注册和 kernel 的实现，Op 是用来描述逻辑上的概念，kernel 实现了具体在物理设备上的运算。所以 Op 更多会关注输入输出的参数名和 shape，计算中需要的属性等等，而 kernel 更关注在不同的设备上的具体计算流程，比如在 cpu 和 gpu 上进行计算就需要不同的实现方式。

所以整体上需要实现一个 Op 只需要注册号 Op，同时完成这个 Op 在不同设备下的 kernel 实现即可。不过因为在模型搭建中还需要考虑更多的问题，比如 Op 需要计算梯度，因为在深度学习中需要通过反向传播算法计算整个计算图中参数的梯度，用户只会显示构建前向图，反向图如何根据前向图进行自动构建等等，所以完成一个 Op 还需要一些额外的步骤，下面我们具体来讲一下每个流程以及其目的。

### Op 注册

第一步需要完成 Op 的注册，即为每个 Op 选择一个唯一的名字，这样当你使用这个名字的时候系统就知道你要调用这个 Op。同时在注册 Op 的时候还需要指定 Op 的输入、输出的参数名，属性的类型和参数名，数据 shape 的推断，参数类型的推断等，最后一个非常重要的作用是设置当前 Op 的 SBP 签名。

SBP 是 oneflow 区别于其他框架的一个重要特性，在 oneflow 中会使用一致性视角来看待所有的张量，用于简化分布式训练，在这个视角下，整个集群会被抽象成一台机器，用户不用关系具体集群的通信细节，只需要关注逻辑上的计算即可，而在逻辑上整个集群和单卡并没有什么区别。在一致性视角下就有了 sbp 等重要概念，要了解这些内容可以查看 [集群的一致性视角 - OneFlow](https://docs.oneflow.org/master/parallelism/02_sbp.html)。

因为一个 Op 不仅需要前向计算，还需要有梯度计算以进行反向传播，所以需要注册 Op 和 Op_grad 分别用于前向和反向。除此之外，还需要额外注册一个 backward Op conf，这个作用是将 Op 和 Op_grad 进行绑定，在静态图中构图时能够自动基于前向的 Op 生成反向的计算图。

#### 注册 Op 的具体实现

前面大致介绍了 Op 的注册流程以及作用，下面以 `leaky_relu` 为例进行简要说明。首先通过宏 `REGISTER_USER_OP` 对 Op 进行注册，在注册过程中会返回一个 `OpRegistry` 对象，可以通过对个对象的方法进行调用来设置 Op 的属性。

**设定输入，输出和属性**

```cpp
REGISTER_USER_OP("leaky_relu")
    .Input("x")
    .Output("y")
    .Attr<float>("alpha")
```

通过 `Input("x")` 和 `Output("y")` 设置了输入和输出的参数名，`Attr<float>("alpha")` 则设置了数据类型是 float 的参数 alpha。

**检查 TensorDesc 的合法性**

```cpp
.SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& x_shape = ctx->InputShape("x", 0);
  Shape* y_shape = ctx->OutputShape("y", 0);
  *y_shape = x_shape;
  return Maybe<void>::Ok();
})
```

`SetTensorDescInferFn` 通过注册回调函数对数据描述进行检查，这里通过输入的 shape 给输出指定 shape 以分配对应的内存。常规的 Op 只需要写一个回调函数，内部会调用 logical 和 physical 的推导设置为同一套，而有一些复杂的 Op 则需要分别写 logical 和 physical 的相同推导。

**设置和推理输出数据类型**

```cpp
.SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
});
```

因为是激活函数，所以设置输出的数据类型和输入一致即可。

**设置 SBP Signature**

```cpp
.SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
})
```

在 sbp signature 的设置中，默认支持 broadcast，如果一下支持其他类型的输入和输出，则需要手工进行设置，比如对于上面的 `leaky_relu` 激活函数，支持在输入和输出的任意维度进行 split，所以可以通过一个在 Axes 上的循环建立不同的 sbp signature。

通过上面的方式可以对前向 Op 和反向 Op 进行注册，最后还需要通过宏 `REGISTER_USER_OP_GRAD` 注册 Op_grad，通过回调函数 `SetGenBackwardOpConfFn` 将前向 Op 和反向 Op 绑定起来，这样在静态图构建前向图时能够自动构建反向图。

### 实现 Kernel 计算

Kernel 是实际计算的控制单元，决定了用什么物理设备进行计算，针对什么样的数据类型以及用哪一种计算方式进行计算，所以我们需要实现在 cpu 和 gpu 下的计算流程，最终实现同一个 Op 可以根据情况使用不同的设备进行计算。

#### Kernel 计算的具体实现

Kernel 的注册和 Op 类似，通过 `REGISTER_USER_KERNEL` 进行注册，在注册之前，需要完成实际的计算过程。

Kernel 都需要继承 `user_op::OpKernel` 这个类，通过 override `Compute` 方法实现具体的计算过程，以 `leaky_relu` 为例。

```cpp
void Compute(user_op::KernelComputeContext* ctx) const override {
  const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
  user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
  const int32_t elem_cnt = x->shape().elem_cnt();
  const float alpha = ctx->Attr<float>("alpha");
  const T* x_ptr = x->dptr<T>();
  T* y_ptr = y->mut_dptr<T>();
  FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : x_ptr[i] * alpha; }
}
```

整个计算过程如下：

1. 获得输入的 tensor "x" 和 输出 tensor "y"，这里的参数名在注册 Op 的时候已经确定；

2. 计算 x 的元素个数；

3. 获得属性 alpha 的值，为一个 float 类型；

4. 获得 tensor x 和 tensor y 的指针用于后续的计算；

5. 遍历所有的元素，根据 leaky_relu 的公式，如果 x[i] > 0 则直接返回 x[i] 的结果，否则返回 alpha \* x[i]。

完成 Kernel 的具体计算流程之后，可以通过下面的方式完成 Kernel 的注册，`SetCreateFn` 可以将这个 Kernel 具体的计算进行绑定，`SetIsMatchedHob` 则接受一些表达式用于对设备和数据类型的判断，比如通过 `REGISTER_CPU_LEAKY_RELU_KERNEL(float)` 则表示在 cpu 设备上，输出 y 的类型是 float 时，使用 `CpuLeakyReluKernel<float>` 进行计算。

```cpp
#define REGISTER_CPU_LEAKY_RELU_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("leaky_relu")                                  \
      .SetCreateFn<CpuLeakyReluKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LEAKY_RELU_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_KERNEL(double)
```

除了需要完成 cpu 的 Kernel 之外，还需要完成 gpu 的 Kernel，整体的注册流程是类似的，不过在 gpu 实现的时候可以充分利用 cuda 编程，这里就不再展开，cuda 相关的内容后续再写成一篇或者几篇文章进行介绍。

### 完成 functional 接口

在 c++ 端完成了 Op 的注册和 Kernel 的实现之后，需要导出到 python 端以及 eager 模式下 autograd engine 进行使用，这是需要利用 functional 进行接口的导出，这样可以通过 `oneflow._C.xxx` 在 python 中对注册接口进行调用。

#### 注册 functional 接口的具体操作

functional 接口的注册主要分为三个步骤：

1. 为接口增加前向和反向的 Functor 实现；

2. 通过 `m.add_functor<impl::MyOpFunctor>("MyOp")` 将 Functor 注册到 Functional Library 中；

3. 在 `functional_api.yaml` 中增加接口的配置文件自动生成接口。

所有的 functor 函数都在 `oneflow/core/functional/impl` 中，被设计成 class 或者是 struct，可以持有一个或是多个 Op。

在 constructor 中将 Op 都构造好，通常只需要声明好 Op 的输入和输出，属性则可以省略。

```c++
class LeakyReluFunctor {
 public:
  LeakyReluFunctor() {
    op_ = CHECK_JUST(one::OpBuilder("leaky_relu").Input("x").Output("y").Build());
  }

 private:
  std::shared_ptr<OpExpr> op_;
};
```

然后实现 `operator()` 接口，在这个接口中完成 Op 的所有计算流程，可以是多个 Op 的组合，oneflow 通过 dispatch op 的机制来调用 Op 下面具体执行计算的 kernel 完成计算流程。

```cpp
Maybe<Tensor> operator()(const std::shared_ptr<one::Tensor>& x, const float& alpha) const {
  MutableAttrMap attrs;
  JUST(attrs.SetAttr<float>("alpha", alpha));
  return OpInterpUtil::Dispatch<one::Tensor>(*op_, {x}, attrs);
}
```

在 `functional_api.yaml` 中增加接口配置文件时，每个接口信息由三个字段组成，示例如下

```yaml
- name: "xxx"
  signature: "R(Args...) => Func"
  bind_python: True or False
```

其中 name 表示导出到 python 接口的名字，signature 指定了接口函数的签名，签名需要和之前定义的 functor 一致，同时 `Func` 作为 signature 的函数名，需要和签名注册到 function library 中的函数名一致，因为需要在 c++ 中使用的是这个函数名。

`bind_python` 表示是否需要为当前的接口生成 python 接口，所有的前向接口都会在 python 搭建模型中用到，所以需要导出到 python，而有的函数不会在 python 中被显示调用，比如求梯度的函数，只会在 c++ 中 autograd 用到，这时就可以不为这种函数增加 python 的接口。

### 注册 eager 求导逻辑

是为了实现 eager 下的 backward op conf，将 eager 过程中的前向 Op 自动绑定反向 Op，所以需要再次注册 eager 下的求导逻辑，主要的代码在 `oneflow/core/autograd/gradient_funcs` 中。

#### eager 求导逻辑的具体实现

首先初始化一个结构体来保存一些属性，这个结构体继承 `AutoGradCaptureState`，比如`requires_grad`, `alpha` 等参数。

```cpp
struct LeakyReluCaptureState : public AutoGradCaptureState {
  bool requires_grad;
  float alpha;
};
```

接着需要实现一个 class 继承 `OpExprGradFunction`，同时需要特例化他的状态结构体 `class LeakyRelu : public OpExprGradFunction<LeakyReluCaptureState>`

接着需要实现下面是个成员函数：

在 `Init` 中完成一些初始化工作，可以根据前向 Op 的 proto 来初始化一个 attrs

```cpp
Maybe<void> Init(const OpExpr& op) override {
  const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  base_attrs_ = MakeAttrMapFromUserOpConf(fw_op_expr->proto());
  return Maybe<void>::Ok();
}
```

接着在 `Capture` 中进行输入的检查，查看是否需要对它求梯度，如果需要的话，就保存一些需要在求梯度的时候使用的内容

```cpp
Maybe<void> Capture(LeakyReluCaptureState* ctx, const TensorTuple& inputs,
                    const TensorTuple& outputs, const AttrMap& attrs) const override {
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  ctx->requires_grad = inputs.at(0)->requires_grad();
  if (!ctx->requires_grad) { return Maybe<void>::Ok(); }

  ComposedAttrMap composed_attrs(attrs, base_attrs_);
  ctx->alpha = JUST(composed_attrs.GetAttr<float>("alpha"));
  ctx->SaveTensorForBackward(inputs.at(0));
  return Maybe<void>::Ok();
}
```

`Apply` 是具体求导的过程，首先可以对后面传回来的梯度 `out_grads` 做一些检查，最后调用在 `functional` 中注册的接口进行梯度的具体计算，然后将结果写回 `in_grads`

```cpp
Maybe<void> Apply(const LeakyReluCaptureState* ctx, const TensorTuple& out_grads,
                  TensorTuple* in_grads) const override {
  CHECK_EQ_OR_RETURN(out_grads.size(), 1);
  in_grads->resize(1);
  if (ctx->requires_grad) {
    const auto& x = ctx->SavedTensors().at(0);
    in_grads->at(0) = JUST(functional::LeakyReluGrad(x, out_grads.at(0), ctx->alpha));
  }
  return Maybe<void>::Ok();
}
```

最后通过 `REGISTER_OP_EXPR_GRAD_FUNCTION` 注册这个 op 的梯度计算逻辑，第一个参数是之前在 user_op 注册中的名字，
第二个参数是刚才定义的类名，比如 `REGISTER_OP_EXPR_GRAD_FUNCTION("leaky_relu", LeakyRelu)`。

## Ending

非常感谢你能看到最后，这篇文章主要是从自己学习的角度出发进行编写，所以整篇文章的目的也是为了梳理自己学习过程中的一些总结，如果这篇文章有任何地方能够帮到你，那就更好了。

最后如果你发现文章有任何纰漏欢迎斧正，如果你有任何建议和意见，也欢迎去评论区留言。

## Reference

- [集群的一致性视角 - OneFlow](https://docs.oneflow.org/master/parallelism/02_sbp.html)

- https://github.com/Oneflow-Inc/oneflow/pull/5854/files

- https://github.com/Oneflow-Inc/oneflow/pull/4130

- https://github.com/Oneflow-Inc/oneflow/pull/5797

- https://github.com/Oneflow-Inc/oneflow/wiki/Functional-Interface

- https://github.com/Oneflow-Inc/oneflow/pull/5329

- https://github.com/Oneflow-Inc/oneflow/pull/6544

- https://cloud.tencent.com/developer/article/1891442
