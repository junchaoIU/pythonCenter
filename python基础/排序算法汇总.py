# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   排序算法汇总
# Author: wujun
# Date:   2021/5/10
# -------------------------------------------------------------------------------

list = [9,8,6,4,7,1,3,5,10,2]

# 冒泡排序
def maopao_sort(list):
    length = len(list)
    for i in range(length):
        Tag = True
        for j in range(1,length-i):
            if(list[j-1]>list[j]):
                list[j-1],list[j]=list[j],list[j-1]
            Tag = False
        if Tag:
            return list
    return list

# 选择排序
def xuanze_sort(list):
    length = len(list)
    for i in range(0,length):
        min = list[i]
        for j in range(i+1,length):
            if(min>list[j]):
                min = list[j]
                list[i],list[j]=list[j],list[i]
    return list

# 插入排序
def charu_sort(list):
    length = len(list)
    for i in range(1,length):
        temp = list[i]
        if(list[i]<list[i-1]):
            index = i
            for j in range(i-1,-1,-1):
                if(list[j]>temp):
                    list[j+1]=list[j]
                    index = j
                else:
                    break;
            list[index]=temp
    return list



if __name__ == '__main__':
    e = charu_sort(list)
    print(e)
