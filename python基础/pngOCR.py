# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   pngOCR
# Author: wujun
# Date:   2021/5/11
# -------------------------------------------------------------------------------
import datetime

from aip import  AipOcr
import  codecs
import os

#读取图片函数
def ocr(path):
    with open(path,'rb') as f:
        return  f.read()

def png2txt(pngPath, txtPath):
    print("{}已经收到，正在处理，请稍后....".format(pngPath))
    startTime_png2txt = datetime.datetime.now()  # 开始时间
    file_list = []
    for file in os.listdir(pngPath):
        if os.path.isfile(os.path.join(pngPath, file)):
            file_list.append(file)
    for pngFile in file_list:
        image = ocr(pngPath+"/"+pngFile)
        # 进程OCR识别
        dict1 = client.basicGeneral(image)
        # print("{}识别完毕".format(pngFile))
        # print(dict1)
        try:
            with codecs.open(txtPath + ".txt", "a", "utf-8") as f:
                for i in dict1["words_result"]:
                    f.write(str(i["words"] + "\r\n"))
        except:
            print("{}出现问题，错误信息：{}".format(pngPath+"/"+pngFile,dict1))
            pass

    endTime_png2txt = datetime.datetime.now()  # 结束时间
    print("{}处理完成".format(txtPath))
    print('转换时间=',(endTime_png2txt - startTime_png2txt).seconds)


if __name__ == '__main__':
    app_id = '23962898'
    api_key = 'EcB1Oav7I2nz6mcVtb5TvDBH'
    secret_key = 'cG3PlkBCLAncczAtaqqTmeRGc60vBodU'
    client = AipOcr(app_id, api_key, secret_key)
    file_list = []
    for file in os.listdir(r'D:\pythonCenter\python基础\png'):
        file_list.append(file)
    file_list = file_list
    print(file_list)
    for bookname in file_list:
        try:
            pngPath = r'D:\pythonCenter\个人小脚本\pdf_to_txt\png\{}'.format(bookname)
            txtPath = r'D:\pythonCenter\个人小脚本\pdf_to_txt\txt\{}'.format(bookname)
            png2txt(pngPath, txtPath)  # 只是转换图片
        except:
            print("{}出现错误！".format(bookname))