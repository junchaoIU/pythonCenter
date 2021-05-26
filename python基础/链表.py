# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   链表
# Author: wujun
# Date:   2021/5/11
# -------------------------------------------------------------------------------

# 节点对象
class Node(object):
    def __init__(self,item):
        self.item = item;
        self.next = None;

# 循环链表对象
class SingleCircleList(object):
    def __init__(self):
        self._head = None

    # 判断链表是否为空：
    def is_empty(self):
        return self._head is None

    # 链表长度
    def length(self):
        if(self.is_empty()):
            return 0;
        else:
            count = 1;
            cur = self._head;
            while(cur.next != self._head):
                count+=1;
                cur = cur.next;
            return count;

    # 遍历链表
    def items(self):
        list = []
        if(self.is_empty()):
            return list;
        else:
            cur = self._head;
            while(cur.next != self._head):
                list.append(cur.item);
                cur = cur.next;
            list.append(cur.item)
            return list;

    # 头部添加节点
    def add(self,item):
        node = Node(item);
        if(self.is_empty()):
            self._head = node;
            node.next = self._head;
        else:
            node.next = self._head;
            cur = self._head;
            while(cur.next != self._head):
                cur = cur.next;
            cur.next = node

        self._head = node

    # 在尾部添加节点
    def append(self,item):
        node = Node(item);
        if(self.is_empty()):
            self._head = node;
            node.next = self._head;
        else:
            cur = self._head;
            while(cur.next != self._head):
                cur = cur.next;
            cur.next = node;
            node.next = self._head;

    # 指定位置添加节点
    def insert(self,index,item):
        node = Node(item);
        if(index <= 0):
            self.add(node);
        elif(index > self.length()-1):
            self.append(node);
        else:
            cur = self._head;
            for i in range(index-1):
                cur = cur.next;
            node.next = cur.next;
            cur.next = node;

    # 删除一个节点
    def delete(self,item):
        node = Node(item);
        # 为第一个节点
        cur = self._head
        if(cur == node):
            curn = cur.next
            while(cur.next != self._head):
                cur = cur.next
            cur.next = curn
            self._head = curn
        else:
            pre = self._head
            while(cur.next != self._head):
                if(cur == node):
                    pre.next = cur.next
                    return True
                else:
                    pre = cur.next
                    cur = cur.next

    # 查询链表元素
    def match(self,item):
        return item in self.items()

if __name__ is "__main__":
    linkList = SingleCircleList()
    print(linkList.is_empty())
    for i in range(5):
        linkList.add(i)
    print(linkList.length())
    print(linkList.items())
    for i in range(5,10):
        linkList.append(i)
    print(linkList.length())
    print(linkList.items())
    linkList.delete(0)
    print(linkList.length())
    print(linkList.items())








