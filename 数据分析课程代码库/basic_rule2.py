#!/usr/bin/env python
#coding:gbk,
import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

a=np.array([3,2,1])#创建数组
print(a)
print(a[:3])
print(a.min())
a.sort()
print(a)
b=np.array([[3,2,1],[4,5,6]])#二维数组
print(b.shape)
print(b.reshape(3,2))
print(b*b)

print(a)
c=np.array([[1],[2],[3]])
print(c)
print(c*a)
print(a@c)

print(b@np.transpose(a))

def f(x):
	x1=x[0]
	x2=x[1]
	return [2*x1-x2**2-1,x1**2-x2-2]
result=fsolve(f,[1,1])
print(result)

x=np.linspace(0,10,1000)#生产指定范围内指定个数的一维数组
y=np.sin(x)+1
z=np.cos(x**2)+1
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(8,4))#设置图像大小
plt.plot(x,y,label='$\sin x+1$',color='red',linewidth=2)
plt.plot(x,z,'b--',label='$\cos x^2+1$')
plt.xlabel('时间time(s)')
plt.ylabel('volt')
plt.title('A simple example')
plt.ylim(0,2.2)#Y轴的范围
plt.legend()#显示图例
plt.show()#显示作图结果

from scipy import linalg
A=np.array([[1,1,7],[2,3,5],[4,2,6]])
b=np.array([2,3,4])
x=linalg.solve(A,b)
print(x)






