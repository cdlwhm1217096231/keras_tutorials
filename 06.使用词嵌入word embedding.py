#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 09:09:58
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$


import os
from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

"""
one-hot编码得到的向量是二进制硬编码的、稀疏的，维度很高(维度等于词汇表中单词的个数)
word embedding采用的是低维的浮点数向量，即密集向量。词嵌入是从数据中学习得到的，常见的词向量维度是256, 512, 1024,词向量可以将更多的信息塞入更低的维度当中。
"""

# 获取词嵌入的方法:
"""
1.在完成任务(如文档分类或情感预测)的同时学习词嵌入。这种情况下，一开始是随机的词向量，然后对这些词向量进行学习，其学习方式与神经网络的权重相同。
2.在不同于待解决问题的机器学习任务上预计算好词嵌入，然后将其加载到模型中，这些词嵌入叫做预训练词嵌入
"""


# 1.利用embedding层学习词嵌入


# 举例将一个Embedding层实例化
# embedding_layer = Embedding(1000, 64)  # 1000代表词汇表中有1000个单词，64代表词向量的维度


# 将Embedding层理解为一个字典，将表示单词的索引映射成密集向量，实质是一种字典查找
"""
Embedding层的输入是一个2D整数张量，形状是(samples, sequence_length),可以嵌入长度可变的序列
一个batch中的所有序列必须具有相同的长度，所以较短的序列用0填充，较长的序列应该截断
Embedding层返回的是(samples, sequence_length, embedding_dimensionality)的3D浮点张量
将一个Embedding层实例化时，它的权重最开始的时候是随机的，在训练过程当中，利用BP逐渐调节这些词向量
改变空间结构，以便下游模型可以使用。
"""

# a.加载IMDB数据，准备用于Embedding层
max_features = 10000  # 找出电影评论中最常见的10000个单词
maxlen = 20   # 只查看每个电影评论句子中的前20个单词
(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features)   # 将数据加载为整数列表
x_train = preprocessing.sequence.pad_sequences(
    x_train, maxlen=maxlen)  # 将整数列表转化为(samples, maxlen)的2D整数张量
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

# b.在IMDB数据上使用Embedding层和分类器


model = Sequential()
# 指定Embedding层的最大输入长度,以便后面将嵌入输入展平，Embedding层的激活形状是(samples, maxlen, 8)
model.add(Embedding(input_dim=10000, output_dim=8, input_length=maxlen))
model.add(Flatten())  # 将3D的嵌入张量展平成shape是(samples, maxlen*8)的2D张量
model.add(Dense(1, activation="sigmoid"))  # 在上面添加分类器
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
model.summary()
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_split=0.2)

"""
上面的模型仅仅将嵌入序列展开并在上面训练一个Dense层，会导致模型对输入序列中的每个单词单独处理，没有考虑单词之间的关系与句子结构，更好的方法是在嵌入序列上添加一个循环层或1D卷积层，将每个序列作为整体来学习特征
"""


# 2.使用预训练的词嵌入

"""
有时候可以用的训练数据很少，以至于只用手头数据无法学习合适特定任务的词嵌入，可以使用预计算的嵌入空间中加载嵌入向量，而不是在解决问题的同时学习词嵌入；常见的预计算的词嵌入数据库，如word2vec、glove等
"""
# a.下载IMDB数据的原始文本,并处理IMDB原始数据
imdb_dir = "./datasets/aclImdb/"
train_dir = os.path.join(imdb_dir, "train")
# 将训练评论转化成字符串列表，每个字符串对应一条评论，将评论的标签转换成labels列表
labels = []
texts = []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(train_dir, label_type)
    for finename in os.listdir(dir_name):
        if finename[-4:] == ".txt":
            f = open(os.path.join(dir_name, finename), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)


# b.对数据进行分词
"""
使用预训练的词嵌入对训练数据很少的问题特别有用，将训练数据限定为前200个样本，因此需要读取200个样本之后，学习对电影评论进行分类
"""
maxlen = 100   # 每条评论取前100个单词后截断原始评论
training_samples = 200  # 200条评论进行训练
validation_samples = 10000  # 10000条评论进行验证
max_words = 10000   # 只考虑所有评论中最常见的10000个单词

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_sequences(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print("Found %s unique tokens." % (len(word_index)))

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print("Shape of data tensor：", data.shape)
print("Shape of label tensor：", labels.shape)


indices = np.arange(data.shape[0])   # 将数据划分成训练集和验证集，但首先要打乱数据，因为原始的数据是好评在前，差评在后
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]


# c.下载glove词嵌入,并对嵌入预处理
glove_dir = "./datasets/glove_dir/"
embedding_index = {}
f = open(os.path.join(glove_dir, "glove.6B.50d.txt"), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embedding_index[word] = coefs
f.close()
print("Found %s word vectors." % len(embedding_index))

# d.准备glove词嵌入矩阵
# 构建一个可以加载到Embedding层中的词嵌入矩阵，它的形状是(max_words, embedding_dim)的矩阵
# 单词索引中索引为i的单词，这个矩阵的元素i就是这个单词对应的词向量
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# e.定义模型
model = Sequential()
model.add(Embedding(input_dim=max_words,
                    output_dim=embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

# f.在模型中加入glove词嵌入
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False   # 冻结Embedding层

# g.训练与评估
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_data=(x_val, y_val))
model.save_weights("pre_trained_glove_model.h5")

# h.绘制结果
import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, "bo", label="Training curve")
plt.plot(epochs, val_acc, "b", label="validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend()
plt.show()


# 在不使用预训练词嵌入的情况下，训练相同的模型
model = Sequential()
model.add(Embedding(input_dim=max_words,
                    output_dim=embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(x_train, y_train, epochs=10,
                    batch_size=32, validation_data=(x_val, y_val))

#  绘制不使用预训练词嵌入的情况下，模型的效果
acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)


plt.plot(epochs, acc, "bo", label="Training curve")
plt.plot(epochs, val_acc, "b", label="validation acc")
plt.title("Training and Validation accuracy(no pre_trained_glove_model)")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss(no pre_trained_glove_model)")
plt.legend()
plt.show()


# i.在测试集上进行分词-----评估模型性能
test_dir = os.path.join(imdb_dir, "test")
# 将训练评论转化成字符串列表，每个字符串对应一条评论，将评论的标签转换成labels列表
labels = []
texts = []
for label_type in ["neg", "pos"]:
    dir_name = os.path.join(test_dir, label_type)
    for finename in os.listdir(dir_name):
        if finename[-4:] == ".txt":
            f = open(os.path.join(dir_name, finename), encoding="utf-8")
            texts.append(f.read())
            f.close()
            if label_type == "neg":
                labels.append(0)
            else:
                labels.append(1)
sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# 在测试集上评估模型
model.load_weights("pre_trained_glove_model.h5")
model.evaluate(x_test, y_test)
