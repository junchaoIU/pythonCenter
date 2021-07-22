# -------------------------------------------------------------------------------
# Description:
# Name:   TCPclient
# Author: Junchao WU
# Date:   2021/7/22
# -------------------------------------------------------------------------------

import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

##建立连接
s.connect(('127.0.0.1', 10240))

# 接收客户端连接成功服务器发来的消息
print(s.recv(1024).decode('utf-8'))

while True:
    data = input('发送给服务器:')
    if len(data) > 0:
        s.send(data.encode('utf-8'))
        print('form sever:{}'.format(s.recv(1024).decode('utf-8')))
s.close()