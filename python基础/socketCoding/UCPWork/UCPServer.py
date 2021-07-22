# -------------------------------------------------------------------------------
# Description:
# Name:   UDPserver
# Author: Junchao WU
# Date:   2021/7/22
# -------------------------------------------------------------------------------


import socket

# 为服务器创建socket并绑定端口  SOCK_DGRAM指定了socket的类型为udp
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

s.bind(('127.0.0.1', 7890))

print('Waiting for data...')
# upd无需监听
while True:
    data, addr = s.recvfrom(1024)
    print('Recevie from {}:{} :{}'.format(addr[0], addr[1], data.decode('utf-8')))
    # send to的另一个参数为客户端socket地址
    s.sendto('信息已成功收到!'.encode('utf-8'), addr)