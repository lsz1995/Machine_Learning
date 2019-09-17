# -*- coding:utf-8 -*-
#贝叶斯公式


#P（A|B） =P(B|A)P(A)/P(B)

#1.贝叶斯拼写检查器

import re,collections

def words(text):
    #转换成小写
    return re.findall('[a-z]+',text.lower())

def train(features):
    model = collections.defaultdict(lambda :1)#https://www.cnblogs.com/duyang/p/5065418.html
    for f  in features:
        model[f]+=1
    return model


NWORDS =train(words(open('./data/big.txt').read()))

#编辑距离  即一个单词通过使用几次插入，删除，交换，替换的操作 变成另外一个单词
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def edits1(word):
    #返回编辑距离为1的集合
    n = len(word)
    # for i in range(n):#删除一个值
    #     print(word[0:i]+word[i+1:])
    # for i in range(n-1):#交换相邻的两个字母
    #     print(word[0:i]+word[i+1]+word[i]+word[i+2:])
    # for  c in alphabet:#随机替换一个字母
    #     for i in range(n):
    #         print(word[0:i]+c+word[i+1:])
    #
    # for  c in alphabet:#随机添加一个字母
    #     for i in range(n):
    #         print(word[0:i] + c + word[i:])
    return set([word[0:i] + word[i + 1:] for i in range(n)] +  # deletion
               [word[0:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(n - 1)] +  # transposition
               [word[0:i] + c + word[i + 1:] for i in range(n) for c in alphabet] +  # alteration
               [word[0:i] + c + word[i:] for i in range(n + 1) for c in alphabet])  # insertion

#编辑距离=2
def edits2(word):
    # a=set()
    # for e1 in edits1(word):
    #     for e2 in edits1(e1):
    #         if e2 in NWORDS:
    #             a.add(e2)
    # print(a)
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

#
def known(words):
    # print(len(words))
    # if len(set(w for w in words if w in NWORDS))!=0:
    #     print(set(w for w in words if w in NWORDS))
    return set(w for w in words if w in NWORDS)
def correct(words):

    candidates = known([words]) or known(edits1(words)) or known(edits2(words)) or [words]
    # print(candidates)
    # for i in candidates:
    #     print(i,NWORDS[i])

    return max(candidates,key=lambda w: NWORDS[w])

if __name__ == '__main__':
    print(correct('mediia'))
