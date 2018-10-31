#coding=utf-8

import pandas
from textrank4zh import TextRank4Sentence, TextRank4Keyword

data_path = '../test_data/ifeng_news.txt.csv'
data_df = pandas.read_csv(data_path)

text = data_df['content'][0]
print(text)

tr_keyword = TextRank4Keyword()

tr_sentence = TextRank4Sentence()

def get_key_sentences(sentences, num=6, sentence_min_len=6):
    """获取最重要的num个长度大于等于sentence_min_len的句子用来生成摘要。

    Return:
    多个句子组成的列表。
    """
    result = []
    count = 0
    sentence_temp = []

    for item in sentences:
        if count >= num:
            break
        if len(item['sentence']) >= sentence_min_len and item['sentence'] not in sentence_temp:
            sentence_temp.append(item['sentence'])
            result.append((item['sentence'], item['index'], item['weight']))
            # result.append(item)
            count += 1
    return result

for text in data_df['content']:
    tr_keyword.analyze(text, lower=True, window=2)
    for keyword in tr_keyword.get_keywords(num=6, word_min_len=1):
        print keyword.word, keyword.weight

    tr_sentence.analyze(text, lower=True, source='all_filters')
    key_sentences = get_key_sentences(tr_sentence.key_sentences, num=5)
    key_sentences.sort(key=lambda s: s[1])
    # sorted(key_sentences, key=lambda s: s[1])
    for s in key_sentences:
        # print(s.index)
        # print(s.weight)
        print(s[0])

