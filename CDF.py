import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class CDF:
    def __init__(self):

        data = np.random.normal(0,10,100)
        res_freq = stats.relfreq(data, numbins=100)
        # 计算结果
        pdf_value = res_freq.frequency
        cdf_value = np.cumsum(res_freq.frequency)
        # 数据1
        F1_arr = np.zeros((100))
        for i in range(0, 5):
            F1_arr[i] = cdf_value[i]
        for i in range(5, 100):
            F1_arr[i] = max(cdf_value[i] - 0.1, np.random.uniform(0, 0.05))
        # 数据2
        F2_arr = np.zeros((100))
        for i in range(0, 2):
            F2_arr[i] = cdf_value[i] + np.random.uniform(0, 0.05)
        for i in range(2, 100):
            F2_arr[i] = max(cdf_value[i] - 0.15, np.random.uniform(0.03, 0.05))
        # 数据3
        F3_arr = np.zeros((100))
        for i in range(0, 2):
            F3_arr[i] = cdf_value[i] + np.random.uniform(0, 0.05)
        for i in range(2, 100):
            F3_arr[i] = max(cdf_value[i] - 0.2, np.random.uniform(0.03, 0.05))
        # 数据4
        # x = np.arange(0,120,3)
        # F4_arr = 0.0033 * x
        F4_arr = np.zeros((100))
        for i in range(0, 2):
            F4_arr[i] = cdf_value[i] + np.random.uniform(0, 0.05)
        for i in range(2, 100):
            F4_arr[i] = max(cdf_value[i] - 0.3, np.random.uniform(0.03, 0.05))


        self.draw_s = np.zeros((5, 100))
        for j in range(100):
            self.draw_s[0][j] = cdf_value[j]
            self.draw_s[1][j] = F1_arr[j]
            self.draw_s[2][j] = F2_arr[j]
            self.draw_s[3][j] = F3_arr[j]
            self.draw_s[4][j] = F4_arr[j]

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
