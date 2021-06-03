# -------------------------------------------------------------------------------
# Description:  使用了 tf.keras训练一个神经网络模型，对运动鞋和衬衫等服装图像进行分类。
# Reference:
# Name:   photos_classification
# Author: wujunchao
# Date:   2021/4/27
# -------------------------------------------------------------------------------

# tensorflow 2.0.0
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

# 从 TensorFlow 中导入和加载 Fashion MNIST 数据
fashion_mnist = keras.datasets.fashion_mnist
# 训练集，测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
图像是28x28的NumPy数组，像素值介于0到255之间。标签是整数数组，介于0到9之间。这些标签对应于图像所代表的服装类：
| 标签 | 类 | | 0 | T 恤/上衣 | | 1 | 裤子 | | 2 | 套头衫 | | 3 | 连衣裙 | | 4 | 外套 | | 5 | 凉鞋 | | 6 | 衬衫 | | 7 | 运动鞋 | | 8 | 包 | | 9 | 短靴 |
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
# (60000, 28, 28) 训练集中有 60,000 个图像，每个图像由 28 x 28 的像素表示
# 60000 同样，训练集中有 60,000 个标签：
# 测试集中有 10,000 个图像

# 预处理数据
# 数据预处理。检查训练集中的第一个图像，像素值处于 0 到 255 之间：
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 将这些值缩小至0到1之间，将这些值除以 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# 验证数据的格式是否正确，显示训练集中的前25个图像，并在每个图像下方显示类名称
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# 设置层
# 该网络的第一层 tf.keras.layers.Flatten 将图像格式从二维数组（28 x 28 像素）转换成一维数组（28 x 28 = 784 像素）。将该层视为图像中未堆叠的像素行并将其排列起来。该层没有要学习的参数，它只会重新格式化数据。
# 展平像素后，网络会包括两个 tf.keras.layers.Dense 层的序列。它们是密集连接或全连接神经层。第一个 Dense 层有 128 个节点（或神经元）。第二个（也是最后一个）层会返回一个长度为 10 的 logits 数组。每个节点都包含一个得分，用来表示当前图像属于 10 个类中的哪一类。
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# 编译模型
# 在准备对模型进行训练之前，还需要再对其进行一些设置。以下内容是在模型的编译步骤中添加的：
# 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
# 在模型训练期间，会显示损失和准确率指标。此模型在训练数据上的准确率达到了 0.91（或 91%）左右。
print("开始训练....")
# 为什么还需要训练多个epochs呢？训练网络时，仅仅将所有数据迭代训练一次是不够的，需要反复训练多次才能使网络收敛。
# 随着epoch的增加，神经网络中权重更新迭代的次数增加，曲线从开始的欠拟合，慢慢进入最佳拟合，epoch继续增加，最后过拟合。
# verbose = 2 为每个epoch输出一行记录
model.fit(train_images, train_labels, epochs=10,verbose =2)

# 评估准确率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc) # verbose = 1 为输出进度条记录

# 进行预测
# 在模型经过训练后，您可以使用它对一些图像进行预测。模型具有线性输出，即 logits。您可以附加一个 softmax 层，将 logits 转换成更容易理解的概率。
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
#预测了测试集中每个图像的标签
predictions = probability_model.predict(test_images)

# 看看第一个预测结果
# 预测结果是一个包含 10 个数字的数组。它们代表模型对 10 种不同服装中每种服装的“置信度”
print(predictions[0])

# 查看哪个标签的置信度值最大
print(np.argmax(predictions[0])) # 因此，该模型非常确信这个图像是短靴，或 class_names[9]。通过检查测试标签发现这个分类是正确的

# 可以将其绘制成图表，看看模型对于全部 10 个类的预测。
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# 验证预测结果
# 查看第 0 个图像、预测结果和预测数组。正确的预测标签为蓝色，错误的预测标签为红色。
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 用模型的预测绘制几张图像。请注意，即使置信度很高，模型也可能出错。
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 使用训练好的模型对单个图像进行预测。
img = test_images[1]
print(img.shape)
# tf.keras 模型经过了优化，可同时对一个批或一组样本进行预测。即便只使用一个图像，也需要将其添加到列表中：
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)

# 预测这个图像的正确标签
predictions_single = probability_model.predict(img)
print(predictions_single)
# 图像
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
# 在批次中获取对我们（唯一）图像的预测：
np.argmax(predictions_single[0]) #2