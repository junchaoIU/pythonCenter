# -------------------------------------------------------------------------------
# Description:
# Name:   TCPserver
# Author: Junchao WU
# Date:   2021/7/22
# -------------------------------------------------------------------------------


import socket
from threading import Thread


def deal(sock, addr):
    print('Accept new connection from {}:{}'.format(addr[0], addr[1]))
    sock.send('与服务器连接成功！'.encode('utf-8'))
    while True:
        data = sock.recv(1024).decode('utf-8')  # 1024为接收数据的最大大小
        print('receive from {}:{} :{}'.format(addr[0], addr[1], data))
        sock.send('信息已成功收到'.encode('utf-8'))


##创建tcp/IPV4协议的socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 为socket绑定端口
s.bind(('127.0.0.1', 10240))
# 监听端口,参数5为等待的最大连接量
s.listen(5)
print("Waiting for connection...")

while True:
    sock, addr = s.accept()
    t1 = Thread(target=deal, args=(sock, addr))
    t1.start()

# 断开与该客户端的连接
sock.close()
s.close()