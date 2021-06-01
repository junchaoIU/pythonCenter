# -------------------------------------------------------------------------------
# Description:  glob 模块提供了方便的文件模式匹配方法
# Reference:
# Name:   globexample
# Author: wujun
# Date:   2021/4/28
# -------------------------------------------------------------------------------

"""
glob 函数支持三种格式的语法：

* 匹配单个或多个字符
? 匹配任意单个字符
[] 匹配指定范围内的字符，如：[0-9]匹配数字。
"""

import glob

print("\n 匹配D:\pythonCenter\数据分析课程代码库下所有py后缀的 文件")
filesList = glob.glob("D:\pythonCenter\数据分析课程代码库\*.py")
print(filesList)