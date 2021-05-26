#coding:gbk,
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # 基于TF-IDF的词频转向量库
from sklearn.cluster import KMeans
import jieba.posseg as pseg


# 中文分词
def jieba_cut(comment):
    seg_list = pseg.cut(comment)  # 精确模式分词[默认模式]
    word_list = [i.word for i in seg_list if i.flag in ['a', 'ag', 'an']] # 只选择形容词追加到列表中
    return word_list


# 读取数据文件
fn = open('comment.txt',encoding='utf8')
comment_list = fn.readlines()  # 读取文件内容为列表
fn.close()

# word to vector
stop_words = ['…', '。', '，', '？', '！', '+', ' ', '、', '：', '；', '（', '）', '.', '-']  # 定义停用词
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=jieba_cut, use_idf=True)  # 创建词向量模型
X = vectorizer.fit_transform(comment_list)  # 将评论关键字列表转换为词向量空间模型

# K均值聚类
model_kmeans = KMeans(n_clusters=3)  # 创建聚类模型对象
model_kmeans.fit(X)  # 训练模型

# 聚类结果汇总
cluster_labels = model_kmeans.labels_  # 聚类标签结果
print(model_kmeans.labels_)
word_vectors = vectorizer.get_feature_names()  # 词向量
print(word_vectors)

word_values = X.toarray()  # 向量值
print(word_values )
comment_matrix = np.hstack(
(word_values, cluster_labels.reshape(word_values.shape[0], 1)))  # 将向量值和标签值合并为新的矩阵
word_vectors.append('cluster_labels')  # 将新的聚类标签列表追加到词向量后面
comment_pd = pd.DataFrame(comment_matrix, columns=word_vectors)  # 创建包含词向量和聚类标签的数据框
print(comment_pd.head(3))  # 打印输出数据框第1条数据

# 聚类结果分析
comment_cluster1 = comment_pd[comment_pd['cluster_labels'] == 2].drop(
'cluster_labels',axis=1)  # 选择聚类标签值为2的数据，并删除最后一列
word_importance = np.sum(comment_cluster1, axis=0)  # 按照词向量做汇总统计
print(word_importance.sort_values(ascending=False)[:5])  # 按汇总统计的值做逆序排序并打印输出前5个词
