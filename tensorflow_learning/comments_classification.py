# -------------------------------------------------------------------------------
# Description:  使用了 tf.keras将影评分为积极（positive）或消极（nagetive）两类。
# Reference:
# Name:   comments_classification
# Author: wujun
# Date:   2021/4/27
# -------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras

import numpy as np

# 下载 IMDB 数据集
# 该数据集已经经过预处理，评论（单词序列）已经被转换为整数序列，其中每个整数表示字典中的特定单词。
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词。为了保持数据规模的可管理性，低频词将被丢弃。

# 了解数据格式。该数据集是经过预处理的：每个样本都是一个表示影评中词汇的整数数组。每个标签都是一个值为 0 或 1 的整数值，其中 0 代表消极评论，1 代表积极评论。
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# 评论文本被转换为整数值，其中每个整数代表词典中的一个单词。
print(train_data[0])

# 电影评论可能具有不同的长度。以下代码显示了第一条和第二条评论的中单词数量。由于神经网络的输入必须是统一的长度，我们稍后需要解决这个问题。
print(len(train_data[0]), len(train_data[1]))

# 将整数转换回单词
# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 使用 decode_review 函数来显示首条评论的文本
print(decode_review(train_data[0]))

# 影评——即整数数组必须在输入神经网络之前转换为张量,填充数组来保证输入数据具有相同的长度
# 由于电影评论长度必须相同，我们将使用 pad_sequences 函数来使长度标准化：
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 看下样本的长度是否相等
print(len(train_data[0]), len(train_data[1]))
# 检查一下首条评论（当前已经填充）
print(train_data[0])

# 构建模型
# 输入形状是用于电影评论的词汇数目（10,000 词）
"""
层按顺序堆叠以构建分类器：

第一层是嵌入（Embedding）层。该层采用整数编码的词汇表，并查找每个词索引的嵌入向量（embedding vector）。这些向量是通过模型训练学习到的。向量向输出数组增加了一个维度。得到的维度为：(batch, sequence, embedding)。
接下来，GlobalAveragePooling1D 将通过对序列维度求平均值来为每个样本返回一个定长输出向量。这允许模型以尽可能最简单的方式处理变长输入。
该定长输出向量通过一个有 16 个隐层单元的全连接（Dense）层传输。
最后一层与单个输出结点密集连接。使用 Sigmoid 激活函数，其函数值为介于 0 与 1 之间的浮点数，表示概率或置信度。
"""
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

# 配置模型来使用优化器和损失函数：
# binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 创建一个验证集
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
# 以 512 个样本的 mini-batch 大小迭代 40 个 epoch 来训练模型。
# 这是指对 x_train 和 y_train 张量中所有样本的的 40 次迭代。在训练过程中，监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）：
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=2)

# 评估模型
results = model.evaluate(test_data,  test_labels, verbose=2)
print(results)

# 创建一个准确率（accuracy）和损失值（loss）随时间变化的图表
history_dict = history.history
history_dict.keys()

# 画图
import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# 损失值（loss）
# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b 代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 准确率（accuracy）
plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()