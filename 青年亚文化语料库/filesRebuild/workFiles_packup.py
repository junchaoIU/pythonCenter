# -------------------------------------------------------------------------------
# Description:
# Reference:
# Name:   work_packup
# Author: wujunchao
# Date:   2021/8/17
# -------------------------------------------------------------------------------
import datetime
import pandas as pd
import os

# check group and files for rebuilt
def get_filesPath(file_dir):
    filesPath = []
    roots = []
    dirses =[]
    for root, dirs, files in os.walk(file_dir):
        dirses.append(dirs)
        roots.append(root)
        for file in files:
            if("_rebuild" in file):
                pass
            else:
                filesPath.append(os.path.join(root, file))
    print("共检索到{}个分组，{}个文件".format(len(dirses[0]),len(filesPath)))
    for i in range(len(dirses[0])):
        dirses[0][i] = file_dir + "\\" + dirses[0][i]
    return dirses[0],filesPath

# merge all rebuild_csv for a group
def groupFiles_merge(grouppath):
    filesList = []
    for root, dirs, files in os.walk(grouppath):
        for file in files:
            filesList.append(os.path.join(root, file))
    print("{}下共找到{}个文件,开始merge......".format(grouppath,len(filesList)))
    mergeCount = 0
    for file in filesList:
        if("_rebuild" in file):
            mergeCount+=1
            df = pd.read_csv(file)
            df.to_csv(grouppath + '\\merge.csv', encoding="utf_8_sig", index=False, header=False, mode='a+')

    print("merge成功，包括{}个rebuild文件，路径为{}".format(mergeCount,grouppath + '\\merge.csv'))

# rebuild all the file in a filepath given
def file_rebuild(filepath,modeNum):
    starttime = datetime.datetime.now()
    data = pd.DataFrame()
    if(modeNum == 0):
        try:
            data = pd.read_csv(filepath,encoding="utf-8")
        except:
            try:
                data = pd.read_csv(filepath, encoding="gbk")
            except:
                print("文件{}无法解析，请手动转化".format(filepath))
    elif(modeNum == 1):
        data = pd.read_excel(filepath)
    if(len(data)>0):
        items = []
        count = 0
        for index, row in data.iterrows():
            if not (pd.isna(row[1])):
                items.append(row)
                count+=1
        if(len(items)>0):
            pditems = pd.DataFrame(items)
            # pditems.columns = ["bulletContent","words"]
            filepath = filepath[:-4]+"_rebuild.csv"
            pditems.to_csv(filepath,index=False)
            endtime = datetime.datetime.now()
            print("处理结束,有效标注数据共{}条，无用数据{}条，耗时{}s，重构文件路径为{}".format(count,{len(data)-count},(endtime - starttime).seconds,filepath))
        else:
            print("文件{}无可用标注数据".format(filepath))
    else:
        print("data为空")

if __name__ == '__main__':
    # check groups and filesList
    groups,filesPathList = get_filesPath(r"C:\Users\WUJO8\Desktop\new")

    # rebuild all files
    for file in filesPathList:
        print("开始读取文件{}".format(file))
        if(file[-4:] == ".csv"):
            file_rebuild(file,0)
        elif(file[-4:] == ".xls" or file[-4:] == "xlsx"):
            file_rebuild(file,1)
        else:
            print("文件{}非需处理的数据类型".format(file))

    # merge files by groups
    print(groups)
    for group in groups:
        groupFiles_merge(group)