import math
import numpy as np
import matplotlib.pyplot as plt

class PDF:
    def __init__(self):
        u = 0   # 均值μ
        sig = math.sqrt(0.2)  # 标准差δ
        x = np.linspace(u - 3*sig, u + 3*sig, 100)   # 定义域
        y = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
        for i in range(len(y)):
            y[i] = y[i] + np.random.uniform(-0.02, 0.01)

        u = 0   # 均值μ
        sig = math.sqrt(0.3)  # 标准差δ
        x = np.linspace(u - 3*sig, u + 3*sig, 100)   # 定义域
        y1 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
        for i in range(len(y)):
            y1[i] = y1[i] + np.random.uniform(-0.02, 0.01)

        u = 0   # 均值μ
        sig = math.sqrt(0.4)  # 标准差δ
        x = np.linspace(u - 3*sig, u + 3*sig, 100)   # 定义域
        y2 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
        for i in range(len(y)):
            y2[i] = y2[i] + np.random.uniform(-0.02, 0.01)

        u = 0   # 均值μ
        sig = math.sqrt(0.5)  # 标准差δ
        x = np.linspace(u - 3*sig, u + 3*sig, 100)   # 定义域
        y3 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
        for i in range(len(y)):
            y3[i] = y3[i] + np.random.uniform(-0.02, 0.01)

        u = 0   # 均值μ
        sig = math.sqrt(0.6)  # 标准差δ
        x = np.linspace(u - 3*sig, u + 3*sig, 100)   # 定义域
        y4 = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2*math.pi)*sig) # 定义曲线函数
        for i in range(len(y)):
            y4[i] = y4[i] + np.random.uniform(-0.02, 0.01)

        self.draw_s = np.zeros((5, 100))

        for j in range(100):
            self.draw_s[0][j] = y[j]
            self.draw_s[1][j] = y1[j]
            self.draw_s[2][j] = y2[j]
            self.draw_s[3][j] = y3[j]
            self.draw_s[4][j] = y4[j]

        x = np.arange(0, 240, 2.4)
        pqsName = ['CD0', 'CD1', 'CD2', 'CD3','CD4']
        lables =['-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
        plt.figure(1)
        for i in range(5):
            y = self.draw_s[i, :]
            pn = pqsName[i]
            la = lables[i]
            plt.plot(x, y, la, label=pn)
        plt.xticks(range(0,250,30))
        plt.xlabel('episodes')
        plt.ylabel('pqs')
        plt.legend(loc='best')  # add legend
        plt.show()

        self.draw_s1 = np.zeros((5, 8))
        for j in range(5):
            for i in range(8):
                self.draw_s1[j][i] = np.average(self.draw_s[j, i * 12: (i + 1) * 12])