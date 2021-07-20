# -------------------------------------------------------------------------------
# Description:
# Name:   task_master
# Author: Junchao WU
# Date:   2021/7/20
# -------------------------------------------------------------------------------

"""
# 如果我们已经有一个通过Queue通信的多进程程序在同一台机器上运行，现在，由于处理任务的进程任务繁重，希望把发送任务的进程和处理任务的进程分布到两台机器上。怎么用分布式进程实现？
# 原有的Queue可以继续使用，但是，通过managers模块把Queue通过网络暴露出去，就可以让其他机器的进程访问Queue了。
"""

# 服务进程负责启动Queue，把Queue注册到网络上，然后往Queue里面写入任务

import queue
import random
from multiprocessing.managers import BaseManager

task_queue = queue.Queue()
result_queue = queue.Queue()

# 发送任务的队列:
def return_task_queue():
    global task_queue
    return task_queue

# 接收结果的队列:
def return_result_queue():
    global result_queue
    return result_queue

# 从BaseManager继承的QueueManager
class QueueManager(BaseManager):
    pass

if __name__ == '__main__':

    # 把两个Queue都注册到网络上, callable参数关联了Queue对象:
    QueueManager.register('get_task_queue', callable=return_task_queue)
    QueueManager.register('get_result_queue', callable=return_result_queue)

    # 绑定本地端口5000, 设置验证码'abc'
    manager = QueueManager(address=('127.0.0.1', 5000), authkey=b'abc')
    # 启动Queue
    manager.start()
    # 获得通过网络访问的Queue对象
    task = manager.get_task_queue()
    result = manager.get_result_queue()
    # 放几个任务进去
    for i in range(10):
        n = random.randint(0, 10000)
        print('Put task %d' % n)
        task.put(n)
    # 从result队列读取结果
    print('Try get results..')
    for i in range(10):
        r = result.get(timeout=10)
        print('Result:%s' % r)

    # 关闭
    manager.shutdown()
    print('master exit.')
