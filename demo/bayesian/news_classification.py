# -*- coding:utf-8 -*-

#贝叶斯新闻分类
#数据来源  搜狗实验室新闻数据

import pandas as pd
import jieba

#######################预处理#####################################
df_news =pd.read_table('./data/data.txt',names=['category','theme','URL','content'],encoding='utf-8')

df_news = df_news.dropna()#删除缺失数据
df_news.tail()
content = df_news.content.values.tolist()#把数据内容页转换成list



######################jieba分词###################

content_S = []
for line in content:
    current_segment = jieba.lcut(line)# 对每篇文章分词
    if len(current_segment) >1 and current_segment!='\r\n':
        content_S.append(current_segment)

# print(content_S[1000])

df_content = pd.DataFrame({'content_S':content_S})
stopwords = pd.read_csv('./data/stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8') #获取停词表

def drop_stopwords(contents,stopwords):
    contents_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        contents_clean.append(line_clean)
    return contents_clean,all_words
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()


contents_clean,all_words = drop_stopwords(contents,stopwords)  # 每篇文章的词  所有的词
# print(len(contents_clean))
# print(len(all_words))
df_content = pd.DataFrame({'contents_clean':contents_clean})#去掉停词后的 结果


df_all_words=pd.DataFrame({'all_words':all_words}) #所有的词
######################jieba分词###################
#######################预处理#####################################

#######################词云#####################################
# import numpy
# words_count=df_all_words.groupby(by=['all_words'])['all_words'].agg({"count":numpy.size})
# words_count=words_count.reset_index().sort_values(by=["count"],ascending=False)
# words_count.head()
#
#
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
# import matplotlib
# matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
#
# wordcloud=WordCloud(font_path="./data/simhei.ttf",background_color="white",max_font_size=80)
# word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
# wordcloud=wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)

#######################词云#####################################

################提取关键词######################
#TF-IDF

import jieba.analyse #工具包
# index = 2400 #随便找一篇文章就行
# content_S_str = "".join(content_S[index]) #把分词的结果组合在一起，形成一个句子
# print (content_S_str) #打印这个句子
#
# print ("  ".join(jieba.analyse.extract_tags(content_S_str, topK=5, withWeight=False)))#选出来5个核心词

################提取关键词######################


df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
label_mapping = {"汽车": 1, "财经": 2, "科技": 3, "健康": 4, "体育":5, "教育": 6,"文化": 7,"军事": 8,"娱乐": 9,"时尚": 0}
df_train['label'] = df_train['label'].map(label_mapping) #构建一个映射方法
# print(df_train.shape)

#####################分训练集  测试集#########################
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
words = [] # 数组形式     把列表转成以‘  ’  隔开的str
for line_index in range(len(x_train)):
    try:
        # x_train[line_index][word_index] = str(x_train[line_index][word_index])
        words.append(' '.join(x_train[line_index]))

    except:
        print (line_index,x_train[line_index])

#####################分训练集  测试集#########################

###############制作词袋模型特征######################
from sklearn.feature_extraction.text import CountVectorizer#词袋模型
# texts=["dog cat fish","dog cat cat","fish bird", 'bird'] #为了简单期间，这里4句话我们就当做4篇文章了
# cv = CountVectorizer() #词频统计
# cv_fit=cv.fit_transform(texts) #转换数据
#
# print(cv.get_feature_names())
# print(cv_fit.toarray())
#
#
# print(cv_fit.toarray().sum(axis=0))


# vec = CountVectorizer(analyzer='word',lowercase = False)
# feature = vec.fit_transform(words)
# print(feature.shape)

vec = CountVectorizer(analyzer='word', max_features=4000,  lowercase = False)
feature = vec.fit_transform(words)#词袋模型特征

# print(feature)
###############制作词袋模型特征######################
#################贝叶斯建模############################

from sklearn.naive_bayes import MultinomialNB #贝叶斯模型
classifier = MultinomialNB()
classifier.fit(feature, y_train)

test_words = []
for line_index in range(len(x_test)):
    try:
        #
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index,x_test[line_index])

print(classifier.score(vec.transform(test_words), y_test))
a = vec.transform(test_words)



for i,j in zip(a,y_test):#预测值与实际值
    print(classifier.predict(i),' ',j)


# print(vec.transform(test_words).toarray())
#################贝叶斯建模############################






###########TF-id  特征######################
from sklearn.feature_extraction.text import TfidfVectorizer

X_test = ['卡尔 敌法师 蓝胖子 小小','卡尔 敌法师 蓝胖子 痛苦女王']

tfidf=TfidfVectorizer()
weight=tfidf.fit_transform(X_test).toarray()
word=tfidf.get_feature_names()
print (weight)
for i in range(len(weight)):
    print (u"第", i, u"篇文章的tf-idf权重特征")
    for j in range(len(word)):
        print (word[j], weight[i][j])

###########TF-idf 特征######################




