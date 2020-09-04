# -*- coding: utf-8 -*-
from __future__ import division
import math
import random
import string
import pickle
import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

starttime = datetime.datetime.now()

flowerLables = {0:'Iris-distance',
                 1:'Iris-light_and_dim',
                2:'hh'}
random.seed(0)
# 生成区间[a, b)内的随机数
def rand(a, b):
    return (b-a)*random.random() + a

# 生成大小 I*J 的矩阵，默认零矩阵 (当然，亦可用 NumPy 提速)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# 函数 sigmoid，这里采用 Symmetrical Sigmoid
def sigmoid(x):
    return math.tanh(x)

# 函数 sigmoid 的派生函数, 为了得到输出 (即：y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    ''' 三层反向传播神经网络 '''
    def __init__(self, ni, nh, no):
        # 输入层、隐藏层、输出层的节点（数）
        self.ni = ni
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no

        # 建立权重（矩阵）
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # 设为随机值
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # 最后建立阈值（矩阵）
        self.ci = makeMatrix(1, self.nh,1)
        self.co = makeMatrix(1, self.no,1)

    def update(self, inputs):
        if len(inputs) != self.ni:
            raise ValueError('与输入层节点数不符！')

        # 激活输入层
        for i in range(self.ni):
            self.ai[i] = inputs[i]

        # 激活隐藏层
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sum - self.ci[0][j]
            self.ah[j] = sigmoid(self.ah[j])

        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sum - self.co[0][k]
            self.ao[k] = sigmoid(self.ao[k])

        # print self.ao
        return self.ao[:]

    def backPropagate(self, targets, N):
        ''' 反向传播 '''
        # if len(targets) != self.no:
        #     raise ValueError('与输出层节点数不符！')

        # 计算输出层的误差
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # 更新隐层到输出层权重和输出层阈值
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change
                self.co[0][k] = self.co[0][k] - N*output_deltas[k]
                #print(N*change, M*self.co[j][k])

        # 更新输入层到隐层权重和隐层阈值
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change
                self.ci[0][j] = self.ci[0][j] - N*hidden_deltas[j]

    def test(self, patterns):
        # count 第一个记录正确的个数，第二个记录识别为第一类的总数，第三个记录识别为第二类的总数
        count = [0]*4
        # All 存放所有的计算结果，并一并返回到train函数，进行绘图
        All = []
        # TP 记录识别每个类正确的总数
        TP = [0] * 3
        # Total 记录所有类真实的总数
        Total = [0] * 3
        # Precision 记录每类识别的精确率
        Precision = [0] * 3
        for p in patterns:
            target = flowerLables[(p[1].index(1))]
            result = self.update(p[0])
            index = result.index(max(result))
            # print(p[0], ':', target, '->', flowerLables[index])
            count[0] += (target == flowerLables[index])
            for k in range(3):
                if p[1].index(1) == k:
                    Total[k] +=1
                if index == k:
                    count[k+1] +=1
                    if p[1].index(1) == k:
                        TP[k] +=1
        accuracy = float(count[0]/len(patterns))
        for i in range(3):
            if count[i+1] == 0:
                Precision[i] = 0
            else:
                Precision[i] = float(TP[i] / count[i+1])
        Recall1 = float(TP[0]/Total[0])
        Recall2 = float(TP[1]/Total[1])
        Recall3 = float(TP[2]/Total[2])
        All.append(accuracy)
        All.append(Precision[0])
        All.append(Precision[1])
        All.append(Precision[2])
        All.append(Recall1)
        All.append(Recall2)
        All.append(Recall3)
        # print('Precision1:%-.9f' % Precision1)
        # print('Precision2:%-.9f' % Precision2)
        # print('Recall1:%-.9f' % Recall1)
        # print('Recall2:%-.9f' % Recall2)
        # print('accuracy: %-.9f' % accuracy)
        return All

    def weights(self):
        print('输入层权重:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('输出层权重:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns,test_data,iterations=1000, N=0.001):
        # N: 学习速率(learning rate)
        Acc = []
        Precision1 = []
        Precision2 = []
        Precision3 = []
        Recall1 = []
        Recall2 = []
        Recall3 = []
        plt.figure()
        plt.xlabel('epho')
        X=range(1,1001)
        y_major_locator = MultipleLocator(0.1)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlim(0,1000)
        plt.ylim(0.2,1.1)
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.backPropagate(targets, N)
            all = self.test(test_data)
            Acc.append(all[0])
            Precision1.append(all[1])
            Precision2.append((all[2]))
            Precision3.append(all[3])
            Recall1.append(all[4])
            Recall2.append(all[5])
            Recall3.append(all[6])
            print(i)
        plt.plot(X,Acc,linewidth='1',label='acc',color='green')
        plt.plot(X,Precision1,label='Precision1',color='blue')
        plt.plot(X,Precision2,label='Precision2',color='pink')
        plt.plot(X,Precision3,label='Precision3',color='black')
        plt.plot(X,Recall1,label='Recall1',color='red')
        plt.plot(X,Recall2,label='Recall2',color='yellow')
        plt.plot(X,Recall3, label='Recall3', color='purple')
        plt.legend(['acc','Precision1','Precision2','Precision3','Recall1','Recall2','Recall3'],loc='right')
        plt.show()
        print('Precision1:%-.9f' % Precision1[999])
        print('Precision2:%-.9f' % Precision2[999])
        print('Precision3:%-.9f' % Precision3[999])
        print('Recall1:%-.9f' % Recall1[999])
        print('Recall2:%-.9f' % Recall2[999])
        print('Recall3:%-.9f' % Recall3[999])
        print('accuracy: %-.9f' % Acc[999])
            # if i % 100 == 0:
            #     error = error /len(patterns)
            #     print('误差 %-.9f' % error)

import pandas as pd
# features 0-255
# labels 256
def iris():
    data = []
    # read dataset
    raw = pd.read_csv('data.csv',header=None)
    raw_data = raw.values
    raw_feature = raw_data[0:,0:256]
    for i in range(len(raw_feature)):
        ele = []
        ele.append(list(raw_feature[i]))
        if i<300:
           ele.append([1,0,0])
        elif i<600:
            ele.append([0,1,0])
        else:
            ele.append([0,0,1])
        data.append(ele)

    # 随机排列data
    random.shuffle(data)
    training = data[0:480]
    test = data[480:]
    nn = NN(256, 19, 3)
    nn.train(training, test, iterations=1000)
    # save weights
    # with open('wi.txt', 'wb') as wif:
    #     pickle.dump(nn.wi, wif)
    # with open('wo.txt', 'wb') as wof:
    #     pickle.dump(nn.wo, wof)

    endtime = datetime.datetime.now()
    print(endtime - starttime)

if __name__ == '__main__':
    iris()