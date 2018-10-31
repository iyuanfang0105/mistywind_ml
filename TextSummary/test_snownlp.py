#coding=utf-8

import pandas
import jieba
from snownlp import SnowNLP



data_path = '../test_data/ifeng_news.txt.csv'
data_df = pandas.read_csv(data_path)

text = data_df['content'][0]

text_segmented = ' '.join(jieba.cut(text, cut_all=False))

print(text_segmented)

# sn_nlp = SnowNLP(text)
#
# print('\n'.join(sn_nlp.words))

# print(sn_nlp.tags)
#
# print(sn_nlp.keywords())

# text = u'''
# 自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。
# 它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
# 自然语言处理是一门融语言学、计算机科学、数学于一体的科学。
# 因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，
# 所以它与语言学的研究有着密切的联系，但又有重要的区别。
# 自然语言处理并不是一般地研究自然语言，
# 而在于研制能有效地实现自然语言通信的计算机系统，
# 特别是其中的软件系统。因而它是计算机科学的一部分。
# '''

s = SnowNLP(text_segmented)

print '\n'.join(s.keywords(20))
print '\n'.join(s.summary(20))
print '\n'.join(s.sentences)

# s = SnowNLP([[u'这篇', u'文章'],
#              [u'那篇', u'论文'],
#              [u'这个']])
# s.tf
# s.idf
# s.sim([u'文章'])# [0.3756070762985226, 0, 0]