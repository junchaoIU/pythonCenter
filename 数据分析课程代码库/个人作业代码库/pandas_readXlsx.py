# -*-coding:utf-8-*-
# @Time    : 2021/03/13 17:14
# @Author  : Wu Junchao
# @Software: PyCharm
# pip --default-timeout=100 install 库名称 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

"""
请用pandas的read_excel方法完成读demo.xlsx的过程，输出要求和案例相同
"""
import pandas as pd  # 导入Pandas库

"""
将日期转换为浮点型
"""
def date_tofloat(data):
    try:
        data = float((data - pd.to_datetime('1899-12-30')).days)
    except:
        data = data
    return data

"""
将整个表中的所有日期转换为浮点型
"""
def sheets_date_tofloat(sheet):
    for i in range(0,len(sheet)):
        sheet.loc[i] = list(map(date_tofloat, sheet.loc[i]))

# 打开文件
xlsx = pd.read_excel('demo.xlsx',header=None,sheet_name=None)
# 查看所有sheet列表
print('All sheets: %s' % list(xlsx))

# 查看sheet1的数据概况
sheet1 = xlsx[list(xlsx)[0]]  # 获得第一张sheet，索引从0开始
sheet1_name = list(xlsx)[0]  # 获得名称
sheet1_cols = len(sheet1.columns.values)   # 获得列数
sheet1_nrows = len(sheet1.index.values)  # 获得行数

print(
'Sheet1 Name: %s\nSheet1 cols: %d\nSheet1 rows: %d' % (sheet1_name, sheet1_cols, sheet1_nrows))

# 将整个表中的所有日期转换为浮点型
sheets_date_tofloat(sheet1)

# 查看sheet1的特定切片数据
sheet1_nrows4 = sheet1.loc[4].values.tolist()  # 获得第5行数据
sheet1_cols2 = sheet1.iloc[:,2].values.tolist()  # 获得第3列数据
cell23 = sheet1.loc[2].values[3]  # 查看第3行第4列数据
cell1=sheet1.loc[1].values[2]
cell2=sheet1.loc[2].values[2]
print(cell1+cell2)
print('Row 4: %s\nCol 2: %s\nCell 1: %s\n' % (sheet1_nrows4, sheet1_cols2, cell23))

# 查看sheet1的数据明细
for i in range(sheet1_nrows):  # 逐行打印sheet1数据
    print(sheet1.loc[i].values.tolist())
print('\n')
for i in range(sheet1_cols):  # 逐列打印sheet1数据
    print(sheet1.iloc[:,i].values.tolist())