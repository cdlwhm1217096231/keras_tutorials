#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-13 22:25:57
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import os
import numpy as np
import string
from keras.preprocessing.text import Tokenizer


"""
用于处理序列的两种基本的深度学习算法,分别是RNN 和 一维的CNN，这些算法的主要应用在:
	1.文档分类和时间序列分类，比如识别文章的主题或书的作者
	2.时间序列对比，如估计两个文档或两支股票行情的相关程度
	3.序列到序列的学习，如机器翻译
	4.情感分析，如将推文或电影评论的情感划分成积极和消极的
	5.时间序列预测，如根据某地最近的天气数据来预测未来天气
深度学习模型不会接收原始文本作为输入，它只能处理数值张量。文本向量化：将文本转换成数值张量的过程，有以下方法：
	1.将文本分隔成单词，并并每个单词转化成一个向量
	2.将文本分隔成字符，并将每个字符转换成一个向量
	3.提取单词或字符的n-gram，并将每个n-gram转化成一个向量。n-gram是多个连续单词或字符的集合
标记(token):将文本分解而获得的单词、字符或n-gram即为标记；
分词(tokenization):将文本分解层标记的过程

将向量与标记相互关联的方法有多种，主要有以下的方法：
	1.对标记做one-hot编码
	2.标记嵌入，通常只用于单词，叫做词嵌入
"""

# 单词与字符的one-hot编码
# 1.单词级的one-hot编码

samples = ["The cat sat on the mat.", "The dog ate my homework."]
token_index = {}  # 构建数据中所有标记的索引
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            # 为每个单词指定唯一的索引，但没有为索引编号0指定单词
            token_index[word] = len(token_index) + 1
max_length = 10
# 对样本进行分词处理，只考虑每个样本前max_length个单词，将分词结果存在results中
# results初始化
results = np.zeros(shape=(len(samples), max_length,
                          max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)  # 根据字典token_index的key即单词名称，获得该单词在词汇表中的索引
        results[i, j, index] = 1.

print(results)
# 2.字符级的one-hot编码
samples = ["The cat sat on the mat.", "The dog ate my homework."]
characters = string.printable   # 所有可打印的ASCII字符
token_index = dict(zip(range(1, len(characters) + 1), characters))
max_length = 50
results = np.zeros(shape=(len(samples), max_length,
                          max(token_index.keys()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1.
print(results)

# 3.keras实现单词级的one-hot编码
samples = ["The cat sat on the mat.", "The dog ate my homework."]
tokenizer = Tokenizer(num_words=100)  # 创建一个分词器，设置词汇表的大小就只有100个单词
tokenizer.fit_on_texts(samples)   # 构建单词的索引
sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转化为整数索引组成的列表
one_hot_results = tokenizer.texts_to_matrix(samples, mode="binary")
word_index = tokenizer.word_index  # 找回单词索引
print(word_index)
print("Found %s unique tokens." % len(word_index))
