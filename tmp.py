'''
from LAC import LAC

# 装载词语重要性模型
lac = LAC(mode='rank')

# 单个样本输入，输入为Unicode编码的字符串
text = u"高管变动:国安国际(00143.HK)委任王淑贞为秘书\
原标题：国安国际(00143.HK)委任王淑贞为秘书    来源：格隆汇\
格隆汇10月31日丨国安国际(00143.HK)公告，王淑贞已获委任为公司公司秘书，自2019年11月1日起生效。\
金融圈'真香'！"
lac_result = lac.run(text)
unque_span=set()
for i in range(1,len(lac_result[0])):
    if lac_result[1][i]==lac_result[1][i-1]:#两个一样，合并成一个span
        lac_result[1][i]='none'
        lac_result[0][i-1]=lac_result[0][i-1]+lac_result[0][i]

for span,span_type in zip(lac_result[0],lac_result[1]):
    if span_type in ['PER','ORG','TIME'] and span not in unque_span:
        unque_span.add((span,span_type))
print(unque_span)
'''
import re
string='123,546,544亿股,204865487584股'
res_span=re.finditer(r"[千数]?[\d亿万]+\.?[\d,]*[亿万w]?[美港欧]?[元%日年月股]",string)
for i in res_span:
    print(i.span(),string[i.span()[0]:i.span()[1]])
