#-*-coding:utf-8-*-
# @Time    : 2020/5/21 2:27
# @Author  : Wu Junchao
# @FileName: wordCloud.py
# @Software: PyCharm

import jieba
with open('创造营.txt', encoding='utf-8') as f:
    mytext = f.read()

print(mytext)

# 词云分析text
from wordcloud import WordCloud
wordcloud = WordCloud(font_path='simhei.ttf').generate(mytext)

# 画图
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()