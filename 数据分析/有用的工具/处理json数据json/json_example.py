# -------------------------------------------------------------------------------
# Description:  JSON (JavaScript Object Notation) 是一种轻量级的数据交换格式，易于人阅读和编写，同时也易于机器解析和生成。
# Reference:
# Name:   json_example
# Author: wujun
# Date:   2021/4/28
# -------------------------------------------------------------------------------

import json
from pprint import pprint

# JSON 与 Python 的转换
StringJson = """
{
    "name": "echo",
    "age": 24,
    "coding skills": ["python", "matlab", "java", "c", "c++", "ruby", "scala"],
    "ages for school": { 
        "primary school": 6,
        "middle school": 9,
        "high school": 15,
        "university": 18
    },
    "hobby": ["sports", "reading"],
    "married": false
}
"""

pprint(StringJson)

print('/n json.loads() (load string) 方法从字符串中读取 JSON 数据')
into = json.loads(StringJson)
pprint(into)
print(type(into))

print('/n 使用 json.dumps() 将一个 Python 对象变成 JSON 对象')
into_json = json.dumps(into)
pprint(into_json)

########################################