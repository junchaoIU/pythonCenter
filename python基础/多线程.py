# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   多线程
# Author: wujun
# Date:   2021/7/20
# -------------------------------------------------------------------------------
import threading
from threading import Lock,Thread
import time,os

'''
    普通创建方式
'''
# def run(n):
#     print('task',n)
#     time.sleep(1)
#     print('2s')
#     time.sleep(1)
#     print('1s')
#     time.sleep(1)
#     print('0s')
#     time.sleep(1)
#
# if __name__ == '__main__':
#     t1 = threading.Thread(target=run,args=('t1',))     # target是要执行的函数名（不是函数），args是函数对应的参数，以元组的形式存在
#     t2 = threading.Thread(target=run,args=('t2',))
#     t1.start()
#     t2.start()

'''
    自定义线程：继承threading.Thread来定义线程类，其本质是重构Thread类中的run方法
'''
# class MyThread(threading.Thread):
#     def __init__(self,n):
#         super(MyThread,self).__init__()   #重构run函数必须写
#         self.n = n
#
#     def run(self):
#         print('task',self.n)
#         time.sleep(1)
#         print('2s')
#         time.sleep(1)
#         print('1s')
#         time.sleep(1)
#         print('0s')
#         time.sleep(1)
#
# if __name__ == '__main__':
#     t1 = MyThread('t1')
#     t2 = MyThread('t2')
#     t1.start()
#     t2.start()

'''
    守护线程
    下面这个例子，这里使用setDaemon(True)把所有的子线程都变成了主线程的守护线程，
    因此当主线程结束后，子线程也会随之结束，所以当主线程结束后，整个程序就退出了。
    所谓’线程守护’，就是主线程不管该线程的执行情况，只要是其他子线程结束且主线程执行完毕，主线程都会关闭。也就是说:主线程不等待该守护线程的执行完再去关闭。
'''
# def run(n):
#     print('task',n)
#     time.sleep(1)
#     print('3s')
#     time.sleep(1)
#     print('2s')
#     time.sleep(1)
#     print('1s')
#
# if __name__ == '__main__':
#     t=threading.Thread(target=run,args=('t1',))
#     t.setDaemon(True)
#     t.start()
#     print('end')

'''
    主线程等待子线程结束
    为了让守护线程执行结束之后，主线程再结束，我们可以使用join方法，让主线程等待子线程执行
'''
# def run(n):
#     print('task',n)
#     time.sleep(2)
#     print('5s')
#     time.sleep(2)
#     print('3s')
#     time.sleep(2)
#     print('1s')
# if __name__ == '__main__':
#     t=threading.Thread(target=run,args=('t1',))
#     t.setDaemon(True)    #把子线程设置为守护线程，必须在start()之前设置
#     t.start()
#     t.join()     #设置主线程等待子线程结束
#     print('end')

'''
    多线程共享全局变量
    线程时进程的执行单元，进程时系统分配资源的最小执行单位，所以在同一个进程中的多线程是共享资源的
'''
# g_num = 100
# def work1():
#     global  g_num
#     for i in range(3):
#         g_num+=1
#     print('in work1 g_num is : %d' % g_num)
#
# def work2():
#     global g_num
#     print('in work2 g_num is : %d' % g_num)
#
# if __name__ == '__main__':
#     t1 = threading.Thread(target=work1)
#     t1.start()
#     time.sleep(1)
#     t2=threading.Thread(target=work2)
#     t2.start()

'''
        由于线程之间是进行随机调度，并且每个线程可能只执行n条执行之后，当多个线程同时修改同一条数据时可能会出现脏数据，
    所以出现了线程锁，即同一时刻允许一个线程执行操作。线程锁用于锁定资源，可以定义多个锁，像下面的代码，当需要独占
    某一个资源时，任何一个锁都可以锁定这个资源，就好比你用不同的锁都可以把这个相同的门锁住一样。
        由于线程之间是进行随机调度的，如果有多个线程同时操作一个对象，如果没有很好地保护该对象，会造成程序结果的不可预期，
    我们因此也称为“线程不安全”。
        为了防止上面情况的发生，就出现了互斥锁（Lock）
'''
# def work():
#     global n
#     lock.acquire()
#     temp = n
#     time.sleep(0.1)
#     n = temp-1
#     lock.release()
#
#
# if __name__ == '__main__':
#     lock = Lock()
#     n = 100
#     l = []
#     for i in range(100):
#         p = Thread(target=work)
#         l.append(p)
#         p.start()
#     for p in l:
#         p.join()


'''
    递归锁：RLcok类的用法和Lock类一模一样，但它支持嵌套，在多个锁没有释放的时候一般会使用RLock类
'''
# def func(lock):
#     global gl_num
#     lock.acquire()
#     gl_num += 1
#     time.sleep(1)
#     print(gl_num)
#     lock.release()
#
#
# if __name__ == '__main__':
#     gl_num = 0
#     lock = threading.RLock()
#     for i in range(10):
#         t = threading.Thread(target=func,args=(lock,))
#         t.start()

'''
    信号量（BoundedSemaphore类）
    互斥锁同时只允许一个线程更改数据，而Semaphore是同时允许一定数量的线程更改数据，比如厕所有3个坑，
    那最多只允许3个人上厕所，后面的人只能等里面有人出来了才能再进去
'''
# def run(n,semaphore):
#     semaphore.acquire()   #加锁
#     time.sleep(3)
#     print('run the thread:%s\n' % n)
#     semaphore.release()    #释放
#
#
# if __name__== '__main__':
#     num=0
#     semaphore = threading.BoundedSemaphore(5)   #最多允许5个线程同时运行
#     for i in range(22):
#         t = threading.Thread(target=run,args=('t-%s' % i,semaphore))
#         t.start()
#     while threading.active_count() !=1:
#         pass
#     else:
#         print('----------all threads done-----------')

'''
    python线程的事件用于主线程控制其他线程的执行，事件是一个简单的线程同步对象，其主要提供以下的几个方法：
        clear将flag设置为 False
        set将flag设置为 True
        is_set判断是否设置了flag
        wait会一直监听flag，如果没有检测到flag就一直处于阻塞状态
    事件处理的机制：全局定义了一个Flag，当Flag的值为False，那么event.wait()就会阻塞，当flag值为True，
    那么event.wait()便不再阻塞
'''
event = threading.Event()
def lighter():
    count = 0
    event.set()         #初始者为绿灯
    while True:
        if 5 < count <=10:
            event.clear()  #红灯，清除标志位
            print("\33[41;lmred light is on...\033[0m]")
        elif count > 10:
            event.set()    #绿灯，设置标志位
            count = 0
        else:
            print('\33[42;lmgreen light is on...\033[0m')

        time.sleep(1)
        count += 1


def car(name):
    while True:
        if event.is_set():     #判断是否设置了标志位
            print('[%s] running.....'%name)
            time.sleep(1)
        else:
            print('[%s] sees red light,waiting...'%name)
            event.wait()
            print('[%s] green light is on,start going...'%name)


# startTime = time.time()
light = threading.Thread(target=lighter,)
light.start()

car = threading.Thread(target=car,args=('MINT',))
car.start()
endTime = time.time()
# print('用时：',endTime-startTime)
