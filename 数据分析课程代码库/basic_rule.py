#!/usr/bin/env python
#coding:gbk,
#python基本操作1
from functools import reduce
a=3
b=a*2
c=a**2#幂运算
print(a,b,c)#打印输出

a,b,c=2,3,"ok"#python支持多重赋值
print(a,b,c)
print(type(c))#显示c的数据类型

s='I like python'#字符串赋值
t=s.split(' ')#
print(t)
p=s+' very much'
print(p)
print(p[2:5])
print(p[0:-1])
print(p[2:])
print(p[:-3])

print('c:\windows\nova')
print(r'c:\windows\nova')

if a==1:
	print(a)
else:
	print('a不等于1')
		
s,k=0,0
while k<101:#0--101连加
	k=k+1
	s=s+k
print(s)

s=0#0--100连加
for k in range(101):#range用来生成连续序列，range(a,b,c)以a为首项，c为公差，且不超过b-1的等差数列
	s=s+k
print(s)

s=0
if s in range(4):
	print('s在0,1,2,3中')
if s not in range(1,4,1):
	print('s不在1,2,3中')

#列表操作
a=[1,2,3,4,5,6]
d=[0,0,0]
del a[5]
b=a
c=a[:]
a[0]=999
print(b)
print(c)

a.append(6)
print(a)
a.extend(d)
print(a)

#列表推导式
squares=[]
for x in range(10):
	squares.append(x**2)
print(squares)
squares2=[x**2 for x in range(10)]
print(squares2)

#元组操作
tup1=(1,2,"a",[d,2,"c"])
print(tup1)
tup2=tuple(d)
print(tup2)
tuple3=tup1+tup2
print(tuple3)

#字典操作
dict1={'list':[1,2,3],1:123,'a':'python','b':(1,2,3)}
for key in dict1:
	print(str(key)+':'+str(dict1[key]))
seq = ('Google', 'Runoob', 'Taobao')
dict2= dict.fromkeys(seq,0)
print("新字典为 : %s" %  str(dict2))

#集合
s=set([1,2,2,3])
q={3,4,5}
print(s)
print(s|q)
print(s&q)
print(s-q)
print(s^q)

#python用def自定义函数	
def add2(x):
	return x+2
print (add2(3))

def add2(x=0,y=0):#参数带默认值
	return[x+2,y+2]#返回一个列表
print(add2())
print(add2(3,4))

def add3(x,y):
	return x+3,y+3#返回多个值
a,b=add3(1,2)
print(a,b)

#函数式编程
f=lambda x:x+2#行内函数
g=lambda x,y:x+y
print(f(3),g(1,2))

a=[1,2,3]
b=map(lambda x: x+2,a)#映射
b=list(b)
print(b)

print(reduce(lambda x,y:x*y, range(1,5+1)))#递归

s=1
for i in range(1,6):
	s=s*i
print(s)

b=filter(lambda x:x>5 and x<8,range(10))#过滤器，用来筛选符合条件的元素
print(list(b))



