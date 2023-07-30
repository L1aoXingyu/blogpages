---
title: VSCode 配置最舒适的深度学习开发环境
pubDatetime: 2022-02-19
description: 记录配置和使用 VSCode 的流程，演示一些深度学习中 debug 的例子
featured: true
tags:
  - vscode,
  - tool
  - development
  - deep learning
ogImage: /assets/vscode-best-practice/cover.png
comments: true
---

> 深度学习往往需要强大的算力和硬件作为支撑，这使得深度学习下的模型和算法大多需要在服务器和集群上运行，区别于传统的程序开发可以在本地的机器上运行，这给深度学习的开发带来了新的挑战。
>
> 在以前，远程开发需要 SSH 登录到服务器上使用命令行和终端进行开发，相比于 Vim 的使用好手可以在终端眼花缭乱地操作代码，大部分人更习惯于 UI 界面和图形化的开发工具，而适应和学习远程开发无疑降低了代码开发效率。而除了代码开发之外，debug 模型和算法也占据着算法开发的一大部分时间，成熟的软件提供了好的 debugger 工具，而远程开发只有命令行和一些 unix 软件，又需要额外学习命令行 debug 的工具链，比如 `ipdb` 和 `gdb`。
>
> 然而现在已经是 2022 年了，远程开发早已变得司空见惯，也有很多软件支持了远程开发的方式，这使得我们不再需要面对黑漆漆的终端和命令行进行开发，也不需要先学习很多复杂的 debugger 工具才能进行模型和算法的开发，极大地增加了我们的开发效率，这篇文章就是总结了笔者经过一段时间探索之后所搭建的最舒适的深度学习开发环境以及一些 debug 的教程。

## Table of contents

## 0x1 选择 VSCode 的原因

对于远程开发来说，有两种方式，一种是本地写代码，然后和远程服务器同步代码；还有一种是直接在服务器上进行开发和写代码。

如果在本地开发代码，不管是用编辑器还是 IDE，最后都需要将代码传到到远程服务器上，而本地环境和远程服务器多少有一些不同，有的时候本地能跑通的代码同步上去可能会有 bug，需要经过二次调试才能在服务器上跑通，而且有一些代码补全功能因为服务器端和本地的环境不同也没法实现，这样的开发效率并不高。

效率更高的开发方式还是直接在远程服务器上进行开发，而炼丹一般来说都是以 Python 语言为主，所以目前有三种方式可以实现远程开发：

1.  SSH 到远程服务器上，然后用 Vim 写代码

    这就是上古时期大神们的开发方式，不过 Vim 的学习成本相对较高，同时 Vim 还需要安装一些插件，进行一定的配置才能比较丝滑地进行开发，否则 Vim 就是一个白板编辑器，而这些步骤相对繁琐，不太适合大部分的炼丹师，容易磨灭大家炼丹的热情；

2.  PyCharm **专业版**

    PyCharm 的专业版提供一个远程同步服务器的功能，同时还可以配置远程的解释器，相当于代码还是在本地写，不过 PyCharm 会帮你自动同步到远程，同时还可以利用远程的解释器进行代码的提示和自动补全，相当于在本地可以享受到远程服务器的开发环境；

    虽然 PyCharm 的使用体验看上去非常方便以及丝滑，这也作为笔者之前一段时间的开发工具，不过使用一段时间之后发现了下面 4 个问题：

    - 专业版需要付费，不像社区版可以免费使用，虽然学生党可以用自己的学校邮箱申请免费使用，不过工作党就只能自己掏钱了，用破解版还是有较大的风险；
    - PyCharm 开发并不是真正在远程进行的，而是通过 SFTP 将本地代码传上去，有的时候可能会出现本地代码没有同步上去的问题，而且有的公司内网需要跳板机才能访问，也就是需要两次 SSH，虽然有解决办法绕过去，但是还是比较麻烦；
    - PyCharm 比较占内存，而一般我们本地开发都是用自己的笔记本，配置一般不会很高，这时候就容易出现卡顿和发热的现象；
    - 最后一个问题是 PyCharm 作为一个 Python 的 IDE，非常适合进行 Python 开发，而对于多语言的开发环境就比较难用了，比如深度学习有时还需要写 C++/CUDA；

3.  VSCode Remote-SSH

    VSCode 也提供了一种远程开发模式可以直接在服务器上写代码，不占用本地机器的内存，同时利用服务器端的开发环境，也不存在开发环境切换的问题，另外有很多社区和官方的插件，配置也比 Vim 简单很多，也能实现代码的自动跳转等功能，对于多语言开发来讲也比较友好，总体来说结合了 Vim 和 PyCharm 各自的优点。

    虽然看上去 VSCode 具备非常大的优点，但是 VSCode 并没有在社区一统天下，它也有一些缺点：

    - Python 的代码编写体验弱于 PyCharm ，代码补全和跳转没有办法做到 PyCharm 那么给力，毕竟 PyCharm 的定位是 Python IDE，而 VSCode 的定位只是一个编辑器；
    - Debug 调试上 PyCharm 不管是 UI 还是 feature 都要比 VSCode 强一些，VSCode 的 debug 还是有一定的学习成本；
    - PyCharm 额外集成了 Django Tools 等工具是 VSCode 没有的；
    - 对于非常大型的项目， VSCode 可能连整个项目的 index 都没法完成，代码跳转等功能会出现间歇性抽风；

上面讲了三种开发方式，一般来说主流的方案都是在 PyCharm 和 VSCode 中选择，上面也列举了不同开发方式各自的优缺点，而笔者最终选择使用 VSCode 作为开发环境也是权衡了上面的利弊。

不过争论到底哪种方式更好其实没有意义，因为最终也不会有结论，大家按照自己的需求进行选择即可。这篇文章也不是为了说服使用 PyCharm 的同学转入 VSCode 的怀抱，更多会讲讲自己是如何配置 VSCode 以及怎么使用 VSCode 能够最大限度发挥其优势，希望能够帮助到正在使用 VSCode 的同学。

## 0x2 深度学习必备插件

使用 VSCode 进行深度学习相关的开发，主要用到的语言就是 Python 和 C++/CUDA，所以下面列举一下使用这两种语言需要用的插件是哪些，另外一些 UI、主题以及一些工具类的插件这里就不在赘述了，毕竟这些内容都是高度个性化的东西，和本文的主题关系并不大，大家可以按照自己的喜好进行这类插件的安装。

### 0x2.1 Python 插件

对于 Python 开发，主要使用的插件是下面两个：

- [Python - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.python): 微软官方 Python 扩展，支持 Python2、3，最重要的是支持 python debugging 功能，这里在后面会具体讲到；
- [Pylance - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance): 微软官方 Python 语言服务器，提供丰富的静态类型检查和自动补全，这会极大提高编程效率；

这两个 Python 插件的配置比较简单，在左边的扩展进行搜索，找到之后直接进行下载即可，基本没有坑。

### 0x2.2 C/C++ 相关插件

- [clangd - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd): LLVM 针对 C/C++ 提供的 Language Server Protocol，可以用它来自动补全代码，实现自动跳转的功能；
- [Native Debug - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=webfreak.debug): 主要是用 gdb 来 debug C/C++ 代码；

在安装完 clangd 的插件之后，可以调出 VSCode 的命令面板，输入 clangd 可以看到自动下载 clangd 的扩展包，可能因为不可描述的服务器网络问题导致扩展包下载失败，可以访问 clangd 的官网进行手工安装。

**注意**：请确保微软官方的 C/C++ 插件已经卸载或者禁用，否则会出现冲突。因为微软的 C++ 插件存在内存泄露和僵尸进程等问题，所以推荐使用上面的插件进行平替。

## 0x3 Debug Python 代码

前面安装的两个 Python 插件，其中 Pylance 提供了 language server protocol 可以自动补全代码，让 Python 的编写效率提高，另外一个插件除了提供语法高亮等内容之外，还提供了非常强大的 debug 功能，如果没有了解的同学可以先阅读一下官方教程 [Debugging configurations for Python apps in Visual Studio Code](https://code.visualstudio.com/docs/python/debugging)，下面举两个例子来说明如何使用 VSCode 高效开发模型。

### 0x3.1 自动定位到错误代码行

有的时候我们会拿到一份可以跑通的代码，但是可能轻微修改完一些超参之后代码就跑不通了，这个时候我们就需要去查看报错信息，而报错信息往往包含很多调用栈，需要在一堆信息中寻找真正的报错位置。

我们可以用下面的报错信息举例，一份训练代码在修改了一个超参之后就出现了下面的报错，通过查看最终的报错信息可以知道好像是某个 BN 层的输入和参数维度对不上，但是网络中有那么多 BN 层，也很难定位是哪个 BN。接着我们可以不断往前找，最终会发现是在 `embedding_head.py` 中的 `124` 行的 bottleneck 调用的时候出现的报错，而 bottleneck 具体在哪里需要继续在代码里面进行查看。

```shell
Traceback (most recent call last):
  File "tools/train_net.py", line 57, in <module>
    args=(args,),
  File "./fastreid/engine/launch.py", line 71, in launch
    main_func(*args)
  File "tools/train_net.py", line 45, in main
    return trainer.train()
  File "./fastreid/engine/defaults.py", line 348, in train
    super().train(self.start_epoch, self.max_epoch, self.iters_per_epoch)
  File "./fastreid/engine/train_loop.py", line 145, in train
    self.run_step()
  File "./fastreid/engine/defaults.py", line 357, in run_step
    self._trainer.run_step()
  File "./fastreid/engine/train_loop.py", line 241, in run_step
    loss_dict = self.model(data)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "./fastreid/modeling/meta_arch/baseline.py", line 112, in forward
    outputs = self.heads(features, targets)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "./fastreid/modeling/heads/embedding_head.py", line 124, in forward
    neck_feat = self.bottleneck(pool_feat)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/container.py", line 139, in forward
    input = module(input)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/modules/batchnorm.py", line 178, in forward
    self.eps,
  File "/home/dev/.local/lib/python3.6/site-packages/torch/nn/functional.py", line 2282, in batch_norm
    input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled
RuntimeError: running_mean should contain 2048 elements not 1024
```

通过上面的方式确实可以定位到错误的地方，但是有没有更高效的 debug 方式呢？下面我们使用 VSCode 进行 debug 来比较两者的差别。根据官网的教程对 debug 进行配置，配置文件一般在 `.vscode/launch.json` 中。

一个参考配置文件如下，`program` 为要运行的程序，`args` 可以将命令行参数传进来，所以下面的 debug 配置转换成 `shell` 命令即为 `python3 tools/train_net.py --config-file configs/Market1501/bagtricks_R50.yml` 命令。

```js
{
  "name": "model training",
  "type": "python",
  "request": "launch",
  "program": "tools/train_net.py",
  "args": [
    "--config-file", "configs/Market1501/bagtricks_R50.yml",
    ],
  "console": "integratedTerminal"
}
```

完成 debug 配置之后，可以通过 `F5` 或者左边的绿色箭头进行 debug，运行之后可以代码会自动停在报错的位置。

<div align=center>
<img src="/assets/vscode-best-practice/debug_model.png" width="800"/>
</div>

仍然是和上面相同的报错，不同在于可以通过 UI 进行展示，不需要去报错里面一层一层的找，另外还有一个好处是可以直观的从左边看到调用栈的信息，同时可以随机进入任何一个调用栈，除此之外还提供了 console 窗口，可以直接打印中间的一些变量结果便于进一步定位问题。

<img src="/assets/vscode-best-practice/debug_python.gif" width="900"/>

### 0x3.2 单步调试代码

除了快速定位问题之外，调试模型的过程中，形状和维度一般是很容易出错的问题，需要不断地检查。下面演示一下在 python debugger 中进行单步调试形状，同时在调试过程中还可以实时查看变量的信息，另外可以不断新增断点，运行到断点位置，可以查看下面的演示。

<img src="/assets/vscode-best-practice/python_step.gif" width="900"/>

## 0x4 Debug C/C++ 代码

c++ 代码一般会使用 `gdb` 进行 debug，`gdb` 也非常强大，可以实现很多功能，不过他的一个缺点就是没有图形界面，在调试复杂代码的时候效率非常低，比如需要不断地打断点，每次 next/step 之后都需要用 list 看一下前后的源码，这种方式虽然也可以达成目标，不过无疑让 debug 的效率降低了。

因为这个原因，市面上也出现了很多搭配 `gdb` 使用的工具，比如 gdbinit, cgdb 等工具，更多详情可以查看这里 [终端调试哪家强？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32843449).

除了上面这些选择之外，还可以搭配 vscode 进行 gdb debug，通过 UI 界面让 gdb 变得更加高效和方便。比如可以在任何位置打断点，进行单步运行，查看变量的值，以及通过条件断点让一个循环在需要的位置停下来，还可以通过 coredump 文件进行 debug，直接访问到异常抛出的位置等等。

下面使用 [gdb Tutorial (cmu.edu)](https://www.cs.cmu.edu/~gilpin/tutorial/) 作为例子，可以去里面查看对应的代码和编译。

首先在 VSCode 中的 `launch.json` 进行下面的配置

```js
{
    "name": "Debug",
    "type": "gdb",
    "request": "launch",
    "target": "./build/gdb_debug",
    "autorun": ["catch throw"],
    "cwd": "${workspaceRoot}",
    "valuesFormatting": "parseText"
}
```

接着点击左边的绿色运行箭头或者是 `F5` 既可以直接开始运行代码，这是会发现代码会在下面的位置抛出异常，左边是函数的调用栈，中间黄色高亮的代码行为运行的位置。

<img src="/assets/vscode-best-practice/gdb_cpp_bug.png" width="900"/>

在下面 console 中可以输入 `gdb` 的命令进行调试，比如下面我们访问 backtrace，然后访问地址 `0x7fffffffd684` 对应的元素，最后发现 `linked list` 在移除 `1` 的时候出现异常。

<img src="/assets/vscode-best-practice/gdb_cpp_console.png" width="900"/>

上面发现当链表在移除 `1` 的时候会抛出异常，所以我们希望程序前面 remove 其他元素的时候能够正常执行，只在 `remove 1` 的时候停下，这个时候就需要借助条件断点，在 gdb 中可以通过下面的命令进行实现 `condition 1 item_to_remove==1`，不过使用 VSCode 会更方便一点，可以通过下面的方式在 UI 上进行操作。

<img src="/assets/vscode-best-practice/gdb_conditional_breakpoint.gif" width="900"/>

当程序停在了我们期望的位置，接下来就可以进行单步调试来确定出问题的地方，如果使用 `gdb` 需要在终端不断地进行 next 指令，同时通过 list 查看具体的代码位置，而在 VSCode 中可以利用现成的 UI 界面，这个时候调试会更加直观。

<img src="/assets/vscode-best-practice/gdb_step.gif" width="900"/>

如果发现自己无法在 VSCode 用鼠标打断点，那么在设置里面搜索 `Debug: Allow Breakpoints Everywhere`，打上勾应该就可以了。

### 0x5 Debug Python/C++ 混合代码

除了调试单独的 Python 和 C++ 代码外，在深度学习中还有一种常见的场景是 Python 和 C++ 混合调试，因为深度学习需要大量的计算，所以大部分计算相关的代码都是用 C++ 或者 CUDA 写的，Python 只是作为前端和接口方便使用。

在调试深度学习模型或者算子的时候，常常需要从 Python 层作为入口，但是最终需要 Debug 到 C++ 层的代码，下面使用 relu 算子举例，如何从进行 Python/C++ 混合 debug。

其实这种类型的 debug 借助 `gdb` 就可以实现，在命令行里面输入下面的指令即可

```shell
# option1
gdb --args python test_relu.py
b <C++ function to break at>
run

# option2
gdb python
b <C++ function to break at>
run relu.py
```

不过用上面的方式在 debug 简单的代码还行，当代码比较复杂之后，还是需要更有效率的 debug 方式，这个时候使用 VSCode 提供的 UI 就可以帮上大忙。

在 VSCode 中按照下面的方式进行配置

```js
{
    "name": "gdb debug",
    "type": "gdb",
    "request": "launch",
    // "env": {"PYTHONPATH": "your python path"}, optional environment variable
    "target": "/home/dev/miniconda/bin/python",
    "arguments": "test_relu.py",
    "autorun": ["catch throw"],
    "cwd": "${workspaceRoot}",
    "valuesFormatting": "parseText"
}
```

按下左边的绿色箭头或者是 `F5` 程序就可以运行了，提前在对应的 C++ 中打上断点，程序最终会在断点位置停下来，然后可以进行单步调试或者是使用 `gdb` 中的命令在 console 中进行想要的操作，非常方便。

<img src="/assets/vscode-best-practice/gdb_python_cpp.gif" width="900"/>

## 0x6 总结

这篇文章中我们介绍了使用 VSCode 的 Python Debugger 进行 Python 代码的调试，以及配合 `gdb` 这个强大的工具对 C++ 和 Python/C++ 混合代码的调试，通过 VSCode 的 UI 以及可交互的 debug 方式可以增加程序开发的效率，希望这篇文章对你有帮助。

如果有任何问题，欢迎在评论去留言。

## 0x7 Reference

- [PyCharm+Docker：打造最舒适的深度学习炼丹炉 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/52827335)
- [Docker+VSCode 配置属于自己的炼丹炉 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/102385239)
- [VSCode+Docker: 打造最舒适的深度学习环境 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/80099904)
- [VSCode 配置 C/C++ 终极解决方案：vs code+clang+clangd+lldb （利用完整的 clang-llvm 工具链） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/398790625?utm_source=pocket_mylist)
- [PyTorch Internals 1：源代码调试方法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/106640360)
- [gdb Tutorial (cmu.edu)](https://www.cs.cmu.edu/~gilpin/tutorial/)
- [终端调试哪家强？ - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/32843449)
