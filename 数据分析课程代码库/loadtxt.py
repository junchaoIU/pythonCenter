#coding:gbk,
#使用Numpy的loadtxt、load、fromfile读取数据


import numpy as np
file_name='numpy_data.txt'
data=np.loadtxt(file_name,dtype='float32',delimiter=' ')
print(data)
print('\n')

write_data=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
np.save('load_data',write_data)
read_data=np.load('load_data.npy')
print(read_data)
print(read_data[:,1])
print('\n')

file_name='numpy_data.txt'
data=np.loadtxt(file_name,dtype='float32',delimiter=' ')
tofile_name='binary'
data.tofile(tofile_name)#丢失数据格式
fromfile_data=np.fromfile(tofile_name,dtype='float32')
print(fromfile_data)
