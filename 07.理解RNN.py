#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-14 11:32:28
# @Author  : cdl (1217096231@qq.com)
# @Link    : https://github.com/cdlwhm1217096231/python3_spider
# @Version : $Id$

import os
import numpy as np

"""
RNN伪代码:
state_t = 0
for input_t in input_sequence:
	output_t = f(input_t, state_t)
	state_t = output_t
-----------------------------------------------
state_t = 0
for input_t in input_sequence:
	output_t = activation(dot(W, input_t) + dot(U, state_t) + b)
	state_t = output_t
"""
# 1.简单RNN 的 numpy实现
timesteps = 100
input_features = 32  # 输入空间的特征维度
output_features = 64  # 输出空间的特征维度

inputs = np.random.random((timesteps, input_features))  # 输入数据：随机噪声
state_t = np.zeros((output_features, ))   # 初始状态初始化为0
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))

successive_outputs = []   # 保存每个输出序列的结果
for input_t in inputs:   # 每个时间序列上的输入为input_t，维度是(input_features, )
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)
print(final_output_sequence.shape)   # 输出的shape是(timesteps, output_features)


# 2.keras中实现RNN

"""
SimpleRNN层是处理批量序列数据，不是像上述的np中一次只能处理单个序列,它接受(batch_size, timesteps, input_features)的输入,SimpleRNN不擅长处理长序列，如文本

SimpleRNN可以在两种不同的模式下运行,根据参数return_sequences参数来控制
	（1）返回每个时间序列的连续输出,形状是(batch_size, timesteps, output_features)
	 (2) 返回最后一个状态的输出, 形状是(bathc_size, output_features)
"""
from keras.layers import SimpleRNN, Embedding
from keras.models import Sequential

model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))   # 只返回最后一个状态的结果
model.summary()
print("--------------------------分隔线-------------------------------")


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))  # 返回完整状态下的序列
model.summary()
print("--------------------------分隔线-------------------------------")


# 为提高网络的表示能力，将多个循环层逐个堆叠有时候也很有用，此时需要将中间层的状态全部返回
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()
print("--------------------------分隔线-------------------------------")


# 3.下面将 上述模型，应用于IMDB电影评论分类

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000   # 作为特征的单词个数
maxlen = 500   # 每条评论只截取前500个单词
batch_size = 32

# a.准备数据
print("加载数据......")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print("train sequences长度:", len(input_train))
print("test sequences长度:", len(input_test))
print("填充序列......")
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)

# b.使用Embedding层和一个SimpleRNN层来训练一个简单的RNN
from keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
history = model.fit(input_train, y_train, epochs=10,
                    batch_size=128, validation_split=0.2)

# c.绘制结果
import matplotlib.pyplot as plt

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and Validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and Validation loss")
plt.legend
plt.show()
