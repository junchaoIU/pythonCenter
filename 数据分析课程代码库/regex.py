#encoding:utf-8
# 导入库
import re
line = ['Cas','a&c','sbmarter','sstan','234697a']
for i in range(len(line)):
	s=re.match('.a',line[i])
	print(s)

print('1*************')
for i in range(len(line)):
	s1=re.match(r'^ss',line[i])
	print(s1)

print('2*************')
for i in range(len(line)):
	s1=re.match(r'a+',line[i])
	print(s1)
print('3*************')
for i in range(len(line)):
	s1=re.match(r'Ca*',line[i])
	print(s1)
print('4*************')
for i in range(len(line)):
	s1=re.match(r's[s|b]m',line[i])
	print(s1)
print('5*************数字')
for i in range(len(line)):
	s1=re.match(r'\d',line[i])
	print(s1)
print('6*************字母')
for i in range(len(line)):
	s1=re.match(r'\D',line[i])
	print(s1)
print('7*************')
for i in range(len(line)):
	s1=re.match(r'a\Wc',line[i])
	print(s1)

tt="Tina is a gOod girl, she is cool, clever, and so on..."
rr=re.compile(r'\woo\w',re.I)
print(rr.findall(tt))

print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.match('com', 'www.runoob.com'))         # 不在起始位置匹配
print('***********')
print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
print(re.search('com', 'www.runoob.com').span())         # 不在起始位置匹配

p=re.compile(r'\d+')
print(p.findall('o1n2m3k4'))

print(re.split('\d+','one1two2three3four4five5'))
print(re.split('\W+', 'runoob, runoob, runoob.'))
print(re.split('(\W+)', 'runoob, runoob, runoob.'))
print(re.split('a+', 'hello world'))

text='python is a kind of computer language, very useful...'
print(re.sub(r'\s+','-',text))


