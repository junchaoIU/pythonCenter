"""
本次的项目实战的总体架构可分为两个步骤：
（1）采用Word2Vector技术去训练词向量；
（2）采用BiLSTM去做特征的表示学习。
"""
import jieba
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

def read_data(data_path):
    senlist = []
    labellist = []
    with open(data_path, "r",encoding='utf-8',errors='ignore') as f:
         for data in f.readlines():
                data = data.strip()
                sen = data.split(",")[0]
                label = data.split(",")[1]
                if sen != "" and (label =="0" or label=="1" or label=="2" ) :
                    senlist.append(sen)
                    labellist.append(label)
                else:
                    pass
    assert(len(senlist) == len(labellist))
    return senlist ,labellist

"""
将一个词映射成一个100维的向量，并且考虑到了上下文的语义。
这里直接将上一部得到的句子列表传给train_word2vec函数就可以了，同时需要定义一个词向量文件保存路径。
模型保存后，以后使用就不需要再次训练，直接加载保存好的模型就可以啦。
"""
def train_word2vec(sentences,save_path):
    sentences_seg = []
    sen_str = "\n".join(sentences)
    # print(sen_str)
    res = jieba.lcut(sen_str)
    # print(res)
    seg_str = " ".join(res)
    # print(seg_str)
    sen_list = seg_str.split("\n")
    for i in sen_list:
        sentences_seg.append(i.split())
    print("开始训练词向量")
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(sentences_seg,
                vector_size=100,  # 词向量维度
                min_count=5,  # 词频阈值
                window=5)  # 窗口大小
    model.save(save_path)
    print("训练成功，词向量文件为{}".format(save_path))
    return model

def generate_id2wec(word2vec_model):
    gensim_dict = Dictionary()
    print(model.wv.key_to_index.keys())
    gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
    w2id = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    print(w2id)
    print(w2id.keys())
    print(model)
    print(model["得"])
    w2vec = {word:model[word] for word in w2id.keys()}  # 词语的词向量
    n_vocabs = len(w2id) + 1
    embedding_weights = np.zeros((n_vocabs, 100))
    for w, index in w2id.items():  # 从索引为1的词语开始，用词向量填充矩阵
        embedding_weights[index, :] = w2vec[w]
    return w2id,embedding_weights

def text_to_array(w2index, senlist):  # 文本转为索引数字模式
    sentences_array = []
    for sen in senlist:
        new_sen = [ w2index.get(word,0) for word in sen]   # 单词转索引数字
        sentences_array.append(new_sen)
    return np.array(sentences_array)

def prepare_data(w2id,sentences,labels,max_len=200):
    X_train, X_val, y_train, y_val = train_test_split(sentences,labels, test_size=0.2)
    X_train = text_to_array(w2id, X_train)
    X_val = text_to_array(w2id, X_val)
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_val = pad_sequences(X_val, maxlen=max_len)
    return np.array(X_train), np_utils.to_categorical(y_train) ,np.array(X_val), np_utils.to_categorical(y_val)

sentences,labels = read_data("data_train.csv")
model =  train_word2vec(sentences,'word2vec.model')
w2id,embedding_weights = generate_id2wec(model)