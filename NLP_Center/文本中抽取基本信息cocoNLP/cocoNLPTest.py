# -------------------------------------------------------------------------------
# Description:  这是中文nlp程序包，可以从文本中提取信息。
# Reference:    https://github.com/fighting41love/cocoNLP
# Name:   cocoNLPTest
# Author: wujunchao
# Date:   2021/4/27
# -------------------------------------------------------------------------------

# 从文本中提取基本信息
"""
抽取email的正则表达式
email_pattern = '^[*#\u4e00-\u9fa5 a-zA-Z0-9_.-]+@[a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*\.[a-zA-Z0-9]{2,6}$'
emails = re.findall(email_pattern, text, flags=0)

抽取phone_number的正则表达式
cellphone_pattern = '^((13[0-9])|(14[0-9])|(15[0-9])|(17[0-9])|(18[0-9]))\d{8}$'
phoneNumbers = re.findall(cellphone_pattern, text, flags=0)

抽取身份证号的正则表达式
IDCards_pattern = r'^([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])$'
IDs = re.findall(IDCards_pattern, text, flags=0)
"""
from cocoNLP.extractor import extractor

ex = extractor()
text = '急寻特朗普，男孩，于2018年11月27号11时在陕西省安康市汉滨区走失。丢失发型短发，...如有线索，请迅速与警方联系：18100065143，132-6156-2938，baizhantang@sina.com.cn 和yangyangfuture at gmail dot com'

# 抽取邮箱
emails = ex.extract_email(text)
print(emails)

# 抽取手机号
cellphones = ex.extract_cellphone(text,nation='CHN')
print(cellphones)

# 抽取身份证号
ids = ex.extract_ids(text)
print(ids)

# 抽取手机归属地、运营商
cell_locs = [ex.extract_cellphone_location(cell,'CHN') for cell in cellphones]
print(cell_locs)

# 抽取地址信息
locations = ex.extract_locations(text)
print(locations)

# 抽取时间点
times = ex.extract_time(text)
print(times)

# 抽取人名
name = ex.extract_name(text)
print(name)

# 从文本中提取短语
from cocoNLP.config.phrase import rake

r = rake.Rake()
# Extraction given the list of strings where each string is a sentence.
r.extract_keywords_from_sentences(['2015年5月11日，“奶茶妹妹”章泽天分别起诉北京搜狐互联网信息服务有限公司、华某（25岁）名誉权纠纷及成某（38岁）名誉权纠纷二案，要求被诉人公开赔礼道歉、恢复名誉、删除相关视频、断开转载该视频的链接，赔偿经济损失、精神损害抚慰金共计170万元。北京市海淀法院已经受理了这两起案件。原告章泽天诉称，她被许多网友称为“奶茶妹妹”，在网络上获得相当的关注度。2014年4月18日，北京搜狐互联网信息服务有限公司的“搜狐视频娱乐播报调查”节目制作并发布了名为“奶茶妹妹恋情或为炒作，百万炒作团队浮出水面”的视频，该段视频捏造包括“奶茶妹妹走红，实为幕后商业策划”、“100万，奶茶妹妹花巨资，请人策划走红”、“奶茶妹妹在清华大学挂科、作弊、想方设法地转学院”等等。华某在上述节目中捏造了大量的对原告的虚假言论，包括声称其就是原告聘请的“幕后推手和炒作专家”，原告曾花100万聘请其为之宣传策划，原告与刘强东的恋情系两者合作的结果等等。'],2,4)

# To get keyword phrases ranked highest to lowest.
ranked_words = r.get_ranked_phrases()

# To get keyword phrases ranked highest to lowest with scores.
ranked_words_score = r.get_ranked_phrases_with_scores()

for ele in ranked_words_score:
    print(ele)
