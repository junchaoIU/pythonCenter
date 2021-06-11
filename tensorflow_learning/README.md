## 什么是TensorFlow
TensorFlow™ 是一个采用 **数据流图**（data flow graphs），用于数值计算的开源软件库。
节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。
它灵活的架构让你可以**在多种平台上展开计算**，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。
TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，**用于机器学习和深度神经网络**方面的研究，但这个系统的通用性使其也可**广泛用于其他计算领域**。

主要优点：
* 灵活性：支持底层数值计算，C++自定义操作符
* 可移植性：从服务器到PC到手机，从CPU到GPU到TPU
* 分布式计算：分布式并行计算，可指定操作符对应计算设备

Tensorflow底层最核心的概念是张量，计算图以及自动微分。

## TensorFlow的结构
TensorFlow的层次结构从低到高可以分成如下五层。

最底层为硬件层，TensorFlow支持CPU、GPU或TPU加入计算资源池。
第二层为C++实现的内核，kernel可以跨平台分布运行。
第三层为Python实现的操作符，提供了封装C++内核的低级API指令，主要包括各种张量操作算子、计算图、自动微分.
如tf.Variable,tf.constant,tf.function,tf.GradientTape,tf.nn.softmax...
如果把模型比作一个房子，那么第三层API就是【模型之砖】。
第四层为Python实现的模型组件，对低级API进行了函数封装，主要包括各种模型层，损失函数，优化器，数据管道，特征列等等。
如tf.keras.layers,tf.keras.losses,tf.keras.metrics,tf.keras.optimizers,tf.data.DataSet,tf.feature_column...
如果把模型比作一个房子，那么第四层API就是【模型之墙】。
第五层为Python实现的模型成品，一般为按照OOP方式封装的高级API，主要为tf.keras.models提供的模型的类接口。
如果把模型比作一个房子，那么第五层API就是模型本身，即【模型之屋】。

### TensorFlow的低阶API
TensorFlow的低阶API主要包括张量操作，计算图和自动微分。
如果把模型比作一个房子，那么低阶API就是【模型之砖】。

在低阶API层次上，可以把TensorFlow当做一个增强版的numpy来使用。
TensorFlow提供的方法比numpy更全面，运算速度更快，如果需要的话，还可以使用GPU进行加速。

张量的操作主要包括张量的结构操作和张量的数学运算。
张量结构操作诸如：张量创建，索引切片，维度变换，合并分割。
张量数学运算主要有：标量运算，向量运算，矩阵运算。另外我们会介绍张量运算的广播机制。
Autograph计算图我们将介绍使用Autograph的规范建议，Autograph的机制原理，Autograph和tf.Module.

### TensorFlow的中阶API
TensorFlow的中阶API主要包括:
* 数据管道(tf.data)
* 特征列(tf.feature_column)
* 激活函数(tf.nn)
* 模型层(tf.keras.layers)
* 损失函数(tf.keras.losses)
* 评估函数(tf.keras.metrics)
* 优化器(tf.keras.optimizers)
* 回调函数(tf.keras.callbacks)

如果把模型比作一个房子，那么中阶API就是【模型之墙】。

### TensorFlow的高阶API
TensorFlow的高阶API主要是tensorflow.keras.models.
tensorflow.keras.models相关的内容：
* 模型的构建（Sequential、functional API、Model子类化）
* 模型的训练（内置fit方法、内置train_on_batch方法、自定义训练循环、单GPU训练模型、多GPU训练模型、TPU训练模型）
* 模型的部署（tensorflow serving部署模型、使用spark(scala)调用tensorflow模型）

### 使用TensorFlow建模
尽管TensorFlow设计上足够灵活，可以用于进行各种复杂的数值计算。
但通常人们使用TensorFlow来实现机器学习模型，尤其常用于实现神经网络模型。
从原理上说可以使用张量构建计算图来定义神经网络，并通过自动微分机制训练模型。
但为简洁起见，一般推荐使用TensorFlow的高层次keras接口来实现神经网络网模型。

使用TensorFlow实现神经网络模型的一般流程包括：
1，准备数据
2，定义模型
3，训练模型
4，评估模型
5，使用模型
6，保存模型。