# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   文本数据建模实验
# Author: wujunchao
# Date:   2021/6/13
# -------------------------------------------------------------------------------

# 准备数据
# imdb数据集的目标是根据电影评论的文本内容预测评论的情感标签。
# 训练集有20000条电影评论文本，测试集有5000条电影评论文本，其中正面评论和负面评论都各占一半。
# 文本数据预处理较为繁琐，包括中文切词（本示例不涉及），构建词典，编码转换，序列填充，构建数据管道等等。

# 在tensorflow中完成文本数据预处理的常用方案有两种
# 第一种是利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Sequence构建文本数据生成器管道。
# 第二种是使用tf.data.Dataset搭配.keras.layers.experimental.preprocessing.TextVectorization预处理层。
# 第一种方法较为复杂，其使用范例可以参考以下文章。
# https://zhuanlan.zhihu.com/p/67697840
# 第二种方法为TensorFlow原生方式，相对也更加简单一些。
# 此处介绍第二种方法

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers,preprocessing,optimizers,losses,metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re,string

train_data_path = "imdb/train.csv"
test_data_path =  "imdb/test.csv"

MAX_WORDS = 10000  # 仅考虑最高频的10000个词
MAX_LEN = 200  # 每个样本保留200个词的长度
BATCH_SIZE = 20


#构建管道
def split_line(line):
    arr = tf.strings.split(line,"\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]),tf.int32),axis = 0)
    text = tf.expand_dims(arr[1],axis = 0)
    return (text,label)

ds_train_raw =  tf.data.TextLineDataset(filenames = [train_data_path]) \
   .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
   .shuffle(buffer_size = 1000).batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames = [test_data_path]) \
   .map(split_line,num_parallel_calls = tf.data.experimental.AUTOTUNE) \
   .batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)


#构建词典
def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
         '[%s]' % re.escape(string.punctuation),'')
    return cleaned_punctuation

vectorize_layer = TextVectorization(
    standardize=clean_text,
    split = 'whitespace',
    max_tokens=MAX_WORDS-1, #有一个留给占位符
    output_mode='int',
    output_sequence_length=MAX_LEN)

ds_text = ds_train_raw.map(lambda text,label: text)
vectorize_layer.adapt(ds_text)
print(vectorize_layer.get_vocabulary()[0:100])


#单词编码
ds_train = ds_train_raw.map(lambda text,label:(vectorize_layer(text),label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text,label:(vectorize_layer(text),label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
