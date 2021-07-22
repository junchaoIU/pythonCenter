# -------------------------------------------------------------------------------
# Description:
# Name:   UDPclient
# Author: Junchao WU
# Date:   2021/7/22
# -------------------------------------------------------------------------------


import socket

# 为服务器创建socket并绑定端口  SOCK_DGRAM指定了socket的类型为udp
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
    data = input('发送给服务器:')
    s.sendto(data.encode('utf-8'), ('127.0.0.1', 7890))

    print('Receive from sever:{}'.format(s.recv(1024).decode('utf-8')))
