# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   word
# Author: wujun
# Date:   2021/6/6
# -------------------------------------------------------------------------------

import pandas as pd
import datetime

class plot_word:
    # 弹幕转换为0，1（判断是否有该热词），获取颜色
    def sum_word(self,data,word):
        color = []
        # 遍历
        for i in range(0,len(data["dm_text"])):
            if(word in data["dm_text"][i]):  # 假如该弹幕中有该热词
                data["dm_text"][i] = 1  # 转换1
                color.append(data["dm_color"][i]) # 颜色添加
            else:  # 否则
                data["dm_text"][i] = 0   # 转换1
        return data,color

    # 时间字符串转换为datetime
    def format_Date1(self,date):
        import datetime
        # 去除秒小数点后的噪音
        date = date.split(".")[0]
        # 转换
        time_format = datetime.datetime.strptime(date,'%H:%M:%S')
        return time_format

    # 弹幕数据读取及预处理
    def read_data(self,data_path):
        data = pd.read_csv(data_path,usecols=["dm_time","dm_color","dm_text"])
        # data["dm_time"] = pd.to_datetime(data["dm_time"], errors='coerce', format='%H:%M:%S')  # 将字符串转换为日期格式
        #遍历转换时间数据
        for i in range(0,len(data["dm_time"])):
            data["dm_time"][i] = self.format_Date1(data["dm_time"][i])
        return data

    def paint_data(self,data):
        # 排序并重置索引
        data = data.sort_values(by="dm_time").reset_index(drop=True)
        # 时间序列
        time = []
        # 每个序列对应词频
        text = []
        for i in range(0,len(data["dm_time"])):
            if(i==0):
                time.append(data["dm_time"][i])
                text.append(data["dm_text"][i])
            else:
                if(data["dm_time"][i]==time[-1]):
                    text[-1]+=data["dm_text"][i]
                else:
                    time.append(data["dm_time"][i])
                    text.append(data["dm_text"][i])
        return time,text

    def paint_word(self,time,text,word):
        import matplotlib.pyplot as plt
        # 画图
        plt.figure()
        plt.title("wordSum")
        plt.plot(time, text, color="m", label=word)  # 设置标签
        plt.ylabel("word_sum")  # y坐标
        plt.xlabel("time")  # x坐标
        plt.grid()
        plt.legend(loc='best')
        plt.rcParams['font.family'] = ['sans-serif']  # 字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.savefig(r'fig/'+word+'.jpg')   # 保存图片

    # 主函数：生成词的时间序列图
    def time_text(self,word):
        # 读取数据并作预处理
        data = self.read_data("testcsv.csv")
        # 将数据切换为时间排序
        data = data.sort_values(by="dm_time")
        # 弹幕转换为0，1（判断是否有该热词），获取颜色
        data,color = self.sum_word(data,word)
        # 时间-词频序列
        time,text = self.paint_data(data)
        # print(time)
        # print(text)
        self.paint_word(time,text,word)

if __name__ == '__main__':
    po = plot_word()
    # 要分析的候选热词表
    data = pd.read_csv("NewWordDiscovery_testcsv.csv_20210607112839.csv",usecols=["word"],encoding='gbk')
    # 遍历词表
    for i in data["word"]:
        po.time_text(i)

    print("over!")
