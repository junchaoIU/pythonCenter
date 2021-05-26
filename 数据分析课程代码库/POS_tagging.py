#coding:gbk,
import jieba.posseg as pseg
import jieba.analyse  # 导入关键字提取库
import pandas as pd
import re
# 读取文本文件
fn = open('./data/article1.txt')  # 以只读方式打开文件
string_data = fn.read()  # 使用read方法读取整段文本
fn.close()  # 关闭文件对象
pattern = re.compile('\t|\n|-|"')  # 建立正则表达式匹配模式
string_data = re.sub(pattern, ' ', string_data)
# 分词+词性标注
words = pseg.cut(string_data)  # 分词
words_list = []  # 空列表用于存储分词和词性分类
for word in words:  # 循环得到每个分词
    words_list.append((word.word, word.flag))  # 将分词和词性分类追加到列表
words_pd = pd.DataFrame(words_list, columns=['word', 'type'])  # 创建结果数据框
print(words_pd.head(5))  # 展示结果前4条

# 词性分类汇总-两列分类
words_gb = words_pd.groupby(['type', 'word'])['word'].count()
print(words_gb.head(5))

# 词性分类汇总-单列分类
words_gb2 = words_pd.groupby('type').count()
words_gb2 = words_gb2.sort_values(by='word', ascending=False)
print(words_gb2.head(5))

# 选择特定类型词语做展示
words_pd_index = words_pd['type'].isin(['n', 'eng'])
words_pd_select = words_pd[words_pd_index]
print(words_pd_select.head(5))


# 关键字提取
print('关键字提取:')
tags_pairs = jieba.analyse.extract_tags(string_data, topK=5, withWeight=True,
                                        allowPOS=['ns', 'n', 'vn', 'v', 'nr'],
                                        withFlag=True)  # 提取关键字标签
tags_list = []  # 空列表用来存储拆分后的三个值
for i in tags_pairs:  # 打印标签、分组和TF-IDF权重
    tags_list.append((i[0].word, i[0].flag, i[1]))  # 拆分三个字段值
tags_pd = pd.DataFrame(tags_list, columns=['word', 'flag', 'weight'])  # 创建数据框
print(tags_pd)  # 打印数据框

