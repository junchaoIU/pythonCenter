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
    def sum_word(self,data,word):
        color = []
        for i in range(0,len(data["dm_text"])):
            if(word in data["dm_text"][i]):
                data["dm_text"][i] = 1
                color.append(data["dm_color"][i])
            else:
                data["dm_text"][i] = 0
        return data,color

    def format_Date1(self,date):
        import datetime
        date = date.split(".")[0]
        time_format = datetime.datetime.strptime(date,'%H:%M:%S')
        return time_format

    def read_data(self,data_path):
        data = pd.read_csv(data_path,usecols=["dm_time","dm_color","dm_text"])
        # data["dm_time"] = pd.to_datetime(data["dm_time"], errors='coerce', format='%H:%M:%S')  # 将字符串转换为日期格式
        for i in range(0,len(data["dm_time"])):
            data["dm_time"][i] = self.format_Date1(data["dm_time"][i])
        return data

    def paint_data(self,data):
        # 排序并重置索引
        data = data.sort_values(by="dm_time").reset_index(drop=True)
        print(data)
        time = []
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
        plt.plot(time, text, color="m", label=word)  # training accuracy类别标签
        plt.ylabel("word_sum")  # y坐标
        plt.xlabel("time")  # x坐标
        plt.grid()
        plt.legend(loc='best')
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.show()

    def time_text(self,word):
        # 读取数据
        data = self.read_data("testcsv.csv")
        # 时间转换
        data = data.sort_values(by="dm_time")
        # 词标转换，获取颜色
        data,color = self.sum_word(data,word)
        # 时间-词频序列
        time,text = self.paint_data(data)
        # print(time)
        # print(text)
        self.paint_word(time,text,word)
        self.paint_rgb(color)

if __name__ == '__main__':
    po = plot_word()
    po.time_text("泪目")