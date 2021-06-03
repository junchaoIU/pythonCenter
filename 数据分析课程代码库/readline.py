
#coding:gbk,
#使用read、readline、readlines读取数据

fn=open('text.txt')
print('光标位置：'+str(fn.tell()))
data1=fn.read()
print('所有数据：\n'+data1)
print('光标位置：'+str(fn.tell()))
line1=fn.readline()
print('第一行数据：\n'+line1)
fn.close

fn=open('text.txt')
print('光标位置：'+str(fn.tell()))
line1=fn.readline()
print('第一行数据：\n'+line1)
line2=fn.readline()
print('第二行数据：\n'+line2+'\n')

fn.close
fn=open('text.txt','r')
line3=fn.readlines()
print('所有数据：')
print(line3)

fn=open('text.txt','a+')
fn.write('\n')
fn.write('line3')
