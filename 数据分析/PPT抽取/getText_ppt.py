# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   getText_ppt
# Author: wujun
# Date:   2021/6/4
# -------------------------------------------------------------------------------
import os
from time import sleep

import win32com
from win32com.client import Dispatch, constants

def read_ppt(name):
    ppt_path = name
    txt_path = "txt/"+name.split(".")[0]+".txt"
    ppt = win32com.client.Dispatch('PowerPoint.Application')
    ppt.Visible = 1
    pptSel = ppt.Presentations.Open(ppt_path)
    # win32com.client.gencache.EnsureDispatch('PowerPoint.Application')

    fout = open(txt_path, "w") #把文字写入该文件
    #get the ppt's pages
    slide_count = pptSel.Slides.Count
    ##print(slide_count)
    for i in range(1,slide_count + 1):
        try:
            fout.write("\n=={}\n".format(i-1))
            shape_count = pptSel.Slides(i).Shapes.Count
            for j in range(1,shape_count + 1):
                try:
                    if pptSel.Slides(i).Shapes(j).HasTextFrame:
                           s = pptSel.Slides(i).Shapes(j).TextFrame.TextRange.Text.strip()
                           if len(s)>0:
                               fout.write("{}\n".format(s))
                except:
                    pass
        except:
            pass
    fout.close()
    ppt.Quit()

def readlist(address):
    return os.listdir(address)

ppt_list = readlist(r"ppt")
txt_list = readlist(r"txt")
for file in ppt_list:
    txt_name = file.split(".")[0]+".txt"
    if(txt_name in txt_list):
        print("{}已扫描".format(file))
        pass
    else:
        try:
            print("{}开始扫描".format(file))
            read_ppt(file)
            sleep(5)
            print("{}扫描成功".format(file))
        except Exception as e:
            print("{}出错，详细信息如下：".format(file))
            print(e)
            # os.remove('txt/'+txt_name)
