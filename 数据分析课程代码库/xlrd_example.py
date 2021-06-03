#coding:gbk,

# 从Excel获取运营数据
# 导入库
import xlrd

# 打开文件
xlsx = xlrd.open_workbook('demo.xlsx')
# 查看所有sheet列表
print('All sheets: %s' % xlsx.sheet_names())

# 查看sheet1的数据概况
sheet1 = xlsx.sheets()[0]  # 获得第一张sheet，索引从0开始
sheet1_name = sheet1.name  # 获得名称
sheet1_cols = sheet1.ncols  # 获得列数
sheet1_nrows = sheet1.nrows  # 获得行数

print(
'Sheet1 Name: %s\nSheet1 cols: %d\nSheet1 rows: %d' % (sheet1_name, sheet1_cols, sheet1_nrows))

# 查看sheet1的特定切片数据
sheet1_nrows4 = sheet1.row_values(4)  # 获得第5行数据
sheet1_cols2 = sheet1.col_values(2)  # 获得第3列数据
cell23 = sheet1.row(2)[3].value  # 查看第3行第4列数据
cell1=sheet1.row(1)[2].value
cell2=sheet1.row(2)[2].value
print(cell1+cell2)
print('Row 4: %s\nCol 2: %s\nCell 1: %s\n' % (sheet1_nrows4, sheet1_cols2, cell23))

# 查看sheet1的数据明细
for i in range(sheet1_nrows):  # 逐行打印sheet1数据
    print(sheet1.row_values(i))
print('\n')
for i in range(sheet1_cols):  # 逐列打印sheet1数据
    print(sheet1.col_values(i))
