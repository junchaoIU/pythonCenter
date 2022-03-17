# -*-coding:utf-8-*-
# @Time    : 2021/01/08 17:43
# @Author  : Wu Junchao
# @FileName: baidubase.py
# @Software: PyCharm
# pip --default-timeout=100 install 库名称 -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

"""
百度智能情感分析平台api
（可定制模型，需上传正向语料和负向语料）
内核基于百度pp飞桨平台下paddlepaddle框架构建的Senta深度情感分析模型算法
"""

from aip import AipNlp

""" 你的 APPID AK SK """
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''

client = AipNlp(APP_ID, API_KEY, SECRET_KEY)

# 对包含主观观点信息的文本进行情感极性类别（积极、消极、中性）的判断，并给出相应的置信度。
"""情感倾向分析 返回数据参数详情
参数名称	是否必选	类型	说明
text	是	string	文本内容（GBK编码），最大2048字节

参数	是否必须	类型	说明
text	是	string	输入的文本内容
items	是	array	输入的词列表
+sentiment	是	number	表示情感极性分类结果, 0:负向，1:中性，2:正向
+confidence	是	number	表示分类的置信度
+positive_prob	是	number	表示属于积极类别的概率
+negative_prob	是	number	表示属于消极类别的概率
"""
text = "苹果是一家伟大的公司"

""" 调用情感倾向分析 """
print("==========================================")
e=client.sentimentClassify(text);
print("文本内容: "+text+"(情感倾向分析)")
print("分析结果json: "+str(e))

print("==========================================")

# 对话情绪识别
# 针对用户日常沟通文本背后所蕴含情绪的一种直观检测，可自动识别出当前会话者所表现出的情绪类别及其置信度，可以帮助企业更全面地把握产品服务质量、监控客户服务质量
# """
# 对话情绪识别接口 请求参数详情
# 参数名称	是否必选	类型	说明
# text	是	string	待识别情感文本，输入限制512字节
# scene	否	string	default（默认项-不区分场景），talk（闲聊对话-如度秘聊天等），task（任务型对话-如导航对话等），customer_service（客服对话-如电信/银行客服等）
# 对话情绪识别接口 返回数据参数详情
#
# 参数	说明	描述
# log_id	uint64	请求唯一标识码
# text	string	输入的对话文本内容
# items	list	分析结果数组
# ++label	string	item的分类标签；pessimistic（强烈负向情绪）、neutral（非强烈负向情绪）
# ++prob	double	item标签对应的概率
# """
# text = "本来今天高高兴兴"
#
# """ 调用对话情绪识别接口 """
# client.emotion(text);
#
# """ 如果有可选参数 """
# options = {}
# options["scene"] = "talk"
#
# """ 带参数调用对话情绪识别接口 """
# print("==========================================")
# e=client.emotion(text, options)
# print("文本内容: "+text+"(对话情绪识别)")
# print("分析结果json: "+str(e))
# print("==========================================")