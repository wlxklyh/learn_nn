# coding=utf-8
import numpy as np
import random
INPUT_DIM_D1 = 1
INPUT_DIM_D2 = 2

OUTPUT_DIM_D1 = 1
OUTPUT_DIM_D2 = 4


class Classifier:
    def __init__(self, s, Tag, hidden_dim=50, reg=0.001, learn_rate=0.001):
        self.samples = s  # 样本数
        self.samples_num = len(s)  # 样本数
        self.Tag = Tag  # 标签数
        
        self.hidden_dim = hidden_dim  # 隐藏层维度 M
        self.reg = reg  # 正则化强度
        self.learn_rate = learn_rate  # 梯度下降的学习率
        
        # 初始化W1，W2，b1，b2
        self.W1 = np.random.randn(INPUT_DIM_D2, self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, OUTPUT_DIM_D2)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.b2 = np.zeros((1, OUTPUT_DIM_D2))
    
    # 前向传播函数 out = x * w + b
    # - x：输入数据（N，D）
    # - w：权重（D，M）
    # - b：偏置（1, M）
    # - out: 输出 (N, M)
    def affine_forward(self, x, w, b):
        x_dot_w = np.dot(x, w)  # (N，D) * (D，M) = (N，M)
        out = x_dot_w + b  # (N, M) + (1, M) = (N, M) 每一行都加一遍(1, M)
        cache = (x, w, b)  # 缓存值，反向传播时使用
        return out, cache
    
    # 反向传播函数 即乘法公式算偏微分
    # ∂Out = X * W + B, ∂Out/∂X = W, ∂Out/∂W = X, ∂Out/∂B = 1
    # dX = dOut * W^T，dW = X * dOut ，dB = 1 * dOut
    # dOut: (N, M)
    # x: (N, D)
    # w: (D, M)
    # b: (1, M)
    # dx: (N, D)
    # dw: (D, M)
    # db: (1, M)
    def affine_backward(self, dOut, cache):
        x, w, b = cache  # 读取缓存
        dx, dw, db = None, None, None  # 返回值初始化
        dx = np.dot(dOut, w.T)  # (N, M) * (D, M)^T = (N,D)
        dw = np.dot(x.T, dOut)  # (N,D)^T * (N, M) = (D,M)
        db = np.sum(dOut, axis=0, keepdims=True)  # 列相加  (N, M) => (1,M)
        return dx, dw, db
    
    def solve(self, loop_n=10000):
        loss = 0
        for j in range(loop_n):
            loss = self.update(j)
            if j % 50 == 0:
                print('loss', loss)
        return loss
    
    def update(self, j):
        sample_index = random.randint(0, self.samples_num-1)
        sample = self.samples[sample_index]
        sample_tag = self.Tag[sample_index]
        # 前向传播
        H, cache1 = self.affine_forward(sample, self.W1, self.b1)  # 第一层前向传播
        H = np.maximum(0, H)  # 激活层 所有小于0的都等于0 ReLU
        relu_cache = H  # 缓存第一层激活后的结果
        
        Y, cache2 = self.affine_forward(H, self.W2, self.b2)  # 第二层前向传播
        
        probs = np.exp(Y)
        probs /= np.sum(probs, axis=1, keepdims=True)       # 计算Softmax的概率 概率最大则是此分类
        
        # 计算loss值
        quadrant_prob =  np.max(sample_tag * probs)  # 对应象限的概率
        loss = -np.log(quadrant_prob)                # Cross Entropy Error 交叉商函数
        
        # 反向传播
        dOut = probs  # 以Softmax输出结果作为反向输出的起点
        dOut-=  sample_tag # 恢复成Y的形状
        
        dh1, dW2, db2 = self.affine_backward(dOut, cache2)  # 反向传播至第二层前 cache2 = H, self.W2, self.b2
        
        dh1[relu_cache <= 0] = 0  # 反向传播至激活层前
        
        dX, dW1, db1 = self.affine_backward(dh1, cache1)  # 反向传播至第一层前 cache1 = self.X, self.W1, self.b1
        
        # 参数更新
        dW1 += self.reg * self.W1  # 正则化 防止过拟合
        dW2 += self.reg * self.W2
        # 负号是因为梯度下降。梯度是指爬升最快的方向，取负就是找下降最快的方向
        self.W2 += -self.learn_rate * dW2
        self.b2 += -self.learn_rate * db2
        self.W1 += -self.learn_rate * dW1
        self.b1 += -self.learn_rate * db1
        return loss
    
    def predict(self, data):
        print("===predict")
        for k in range(data.shape[0]):
            # 前向传播
            H, fc_cache = self.affine_forward(data[k], self.W1, self.b1)
            # 激活
            H = np.maximum(0, H)
            # 前向传播
            Y, _ = self.affine_forward(H, self.W2, self.b2)
            # Softmax
            probs = np.exp(Y)
            probs /= np.sum(probs, axis=1, keepdims=True)
            result = np.argmax(probs) + 1
            print(data[k, :], "所在的象限为", result)


np.random.seed(1)
np.set_printoptions(suppress=True)
# (N, D)
Sample = np.array([
    [[2, 1]],
    [[-1, 1]],
    [[-1, -1]],
    [[1, -1]],
])

# (T, 1)
Tag = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1],
])

c = Classifier(Sample, Tag, 50)
c.solve()  # 训练

test = np.array([
    [[2, 2]],
    [[-2, 2]],
    [[2, 1]],
    [[-1, 1]],
    [[-1, -1]],
    [[1, -1]],
])
c.predict(test)  # 预测test的每个坐标属于哪个象限(类别)
