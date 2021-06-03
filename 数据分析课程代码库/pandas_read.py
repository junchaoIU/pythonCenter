
#coding:gbk,
#使用Pandas的read_csv、read_fwf、read_table读取数据

import pandas as pd  # 导入Pandas库

csv_data = pd.read_csv('csv_data.csv', names=['col1', 'col2', 'col3', 'col4', 'col5'])  # 读取csv数据
print(csv_data)  # 打印输出数据
print (type(csv_data))


fwf_data = pd.read_fwf('fwf_data', widths=[5, 3, 6, 6],
                       names=['col1', 'col2', 'col3', 'col4'])  # 读取csv数据
print(fwf_data)  # 打印输出数据

table_data = pd.read_table('table_data.txt', sep=';',
                           names=['col1', 'col2', 'col3', 'col4', 'col5'])  # 读取txt数据
print(table_data)  # 打印输出数据


