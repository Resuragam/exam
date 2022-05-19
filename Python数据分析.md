# Python 数据分析

## 一、Python数据分析概述

### 1. 数据分析的概念、流程以及应用

#### ==Ⅰ -数据分析的概念==

> **数据分析**
>
> ① 概念：数据分析是指选用适当的分析方法对收集来的大量数据进行分析、提取有用的信息和形成结论，对数据加以详细研究和概况总结的过程
>
> ② 广义的数据分析包括**狭义的数据分析**和**数据挖掘**。
>
> > 数据挖掘是指从大量的、不完全的、有噪音的、模糊的、随机的实际应用数据当中，通过应用的聚类模型、分类模型、回归和关联规则等技术，挖掘潜在价值的过程

<hr>

#### Ⅱ -数据分析的流程

> ① 需求分析
>
> ② 数据获取
>
> ③ 数据预处理
>
> ④ 分析与建模
>
> ⑤ 模型评价与优化
>
> ⑥ 部署

<hr>

#### Ⅲ -数据分析的应用

> ① 客户与营销分析
>
> ② 业务流程优化
>
> ③ 完善执法
>
> ④ 网络安全
>
> ⑤ 优化机器和设备性能
>
> ⑥ 改善日常生活
>
> ⑦ 医疗卫生与生命科学

<hr>

### 2. 数据分析工具

#### Ⅰ -常用工具

>① **Python**
>
>② R语言
>
>③ Matlab

<hr>

#### ==Ⅱ -Python数据分析==

##### ① 优势

> * 语法精练简单
>
> * 功能强大的库
>
> * 功能强大
>
> * 不仅适用于研究和原型构建，同时也适用于构建生产系统
>
> * Python是一门胶水语言。Python可以通过多种方式与其他语言组织粘连，调用其他语言程序。

##### ② ==常见的类库==

> * **NumPy**
>
> * **Pandas**
>
> * **Matplotlib**
>
> * **Sklearn**
>
> * 其他

<hr>

## 二、NumPy数值计算

### 1. NumPy 多维数组

#### ==Ⅰ-数组创建==

##### ① array()函数创建

>* 一维列表
>
>  ~~~python
>  a1 = np.array([1, 2, 3, 4, 5])
>  print(a1)
>  # [1 2 3 4 5]
>  ~~~
>
>* 二维列表
>
>  ~~~python
>  a2 = np.array([[1, 2, 3], [4, 5, 6]])
>  print(a2)
>  #[[1 2 3][4 5 6]]
>  ~~~
>
>* 字符串
>
>  ~~~python
>  a3 = np.array('python')
>  print(a3)
>  # python
>  ~~~
>
>* 元组
>
>  ~~~python
>  a4 = np.array((1, 2, 3))
>  print(a4)
>  #[1 2 3]
>  ~~~
>
>* 字典
>
>  ~~~python
>  a5 = np.array({'zhang': 12, 'huang': 25})
>  print(a5)
>  #{'zhang': 12, 'huang': 25}
>  ~~~

<hr>

##### ② 创建特殊的数组

> | 函数                         | 描述                     |
> | ---------------------------- | ------------------------ |
> | ones()         ones_like()   | 指定形状的全1数组        |
> | zeros()        zeros_like()  | 指定形状的全0数组        |
> | empty()      empty_like()    | 指定形状的没有具体值数组 |
> | eye()            indentity() | N*N单位矩阵              |
>
> ~~~python
> import numpy as np
> b1 = np.ones((3, 4))
> print(b1)
> 
> b2 = np.ones_like(b1)
> print(b2)
> 
> b4 = np.eye(4)
> print('b4=', b4)
> ~~~
>
> ![1652935831589](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652935831589.png)

<hr>

##### ③ 从数值范围创建数组

###### （1） `arange()`函数

> `arange()`函数根据`start`和`stop`指定的范围`[start,stop)`，根据步长`step`，生成数组对象
>
> ~~~python
> # np.arange(start,stop,step,dtype)
> # dtype指定数组返回的数据类型
> c1 = np.arange(1,5,2)
> print(c1)
> # [1 3]
> ~~~

###### （2）`linspace()`函数

> `linspace()`函数生成一个等差数列构成的一维数组
>
> 参数`endPoint`是布尔类型，控制数组是否包括`stop`值
>
> 参数`retstep`是布尔类型，控制返回结果是否显示间距
>
> ~~~~python
> # np.linspace(start,stop,num,endpoint,retstep,dtype)
> c2 = np.linspace(1, 50, 50, True, True)
> print(c2)
> ~~~~
>
> ![1652936711826](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652936711826.png)

###### （3）`logspace()`函数

> `logspace()`函数生成一个==对数运算的等比数列==构成的一维数组
>
> 参数`endPoint`是布尔类型，控制数组是否包括`stop`值
>
> 参数`base`是对数的底数
>
> ~~~~python
> # np.logspace(start,stop,num,endpoint,base,dtype)
> c3 = np.logspace(1, 10, 10, True, 10)
> print(c3)
> ~~~~
>
> ![1652936711826](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652936711826.png)

<hr>

##### ④ 使用`asarray()`函数创建数组

> `asarray()`函数可以把列表，元组等任意形式参数转化为`NumPy`数组
>
> `tolist()`函数可以转化为`Python`列表
>
> ~~~python
> d1 = [1, 3, 5, 7, 9]
> d2 = np.asarray(d1)
> print(d2)
> #[1,3,5,7,9]
> ~~~

<hr>

##### ⑤ 创建随机数数组

`Numpy`的随机数函数在`Numpy.random`模块中，==因此调用相关函数需要加上**`random`**==`

###### （1）`rand()`函数

> `rand()`函数生成**指定形状**，服从分布在[0,1)之间的随机数数组
>
> ~~~python
> a = np.random.rand(2, 2, 3)
> print(a)
> ~~~
>
> ![1652938522762](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652938522762.png)

###### （2）`uniform()`函数

> `uniform()`函数生成**指定形状**内，服从**指定区间**之间的随机数数组
>
> ~~~python
> b = np.random.uniform(4, 5, (2, 2, 2))
> print(b)
> ~~~
>
> ![1652938866976](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652938866976.png)

###### （3）`randn()`函数

> `randn()`函数生成**指定形状**内，服从**标准正态分布**的随机数数组

###### （4）`normal()`函数

> `normal()`函数生成**指定形状**内，服从**指定正态分布**的随机数数组
>
> 默认是标准正态分布，`size`不设置返回符合指定正态分布的一个随机数
>
> ~~~python
> # np.random.normal(loc,scale,size)
> c = np.random.normal(1,2,(2,2))
> print(c)
> ~~~

###### （5）`random()`函数

> `random()`函数生成**指定形状**内，[0,1)之间均匀抽样的数组

###### （6）`randint()`函数

> `randint()`函数生成**指定形状**内，**指定区间**之间均匀抽样的`int`型数组
>
> ~~~python
> # np.random.randint(low,high,size,dept)
> c = np.random.randint(1,2,(2,2))
> print(c)
> ~~~
>
> ![1652939799933](C:\Users\HHY\AppData\Roaming\Typora\typora-user-images\1652939799933.png)

<hr>



