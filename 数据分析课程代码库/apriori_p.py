#!/usr/bin/env python
#coding:gbk,
#关联规则

import sys
sys.path.append('./apri')
import pandas as pd
import apriori

# 定义数据文件
fileName = 'menu_orders.xls'
#fileName='menu_orders.xls'
# 通过调用自定义的apriori做关联分析
minS = 0.1  # 定义最小支持度阀值
minC = 0.1  # 定义最小置信度阀值
dataSet = apriori.createData(fileName)  # 获取格式化的数据集
print(dataSet)
L, suppData = apriori.apriori(dataSet, minSupport=minS)  # 计算得到满足最小支持度的规则
rules = apriori.generateRules(fileName, L, suppData, minConf=minC)  # 计算满足最小置信度的规则
# 关联结果报表评估
model_summary = 'data record: {1} \nassociation rules count: {0}'  # 展示数据集记录数和满足阀值定义的规则数量
print (model_summary.format(len(rules), len(dataSet)))  # 使用str.format做格式化输出
df = pd.DataFrame(rules, columns=['item1', 'item2', 'instance', 'support', 'confidence', 'lift'])  # 创建频繁规则数据框
df_lift = df[df['lift'] > 1.0]  # 只选择提升度>1的规则
print(df_lift.sort_values('instance', ascending=False))  # 打印排序后的数据框



