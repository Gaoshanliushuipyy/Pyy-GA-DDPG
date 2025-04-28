
import numpy as np
import matplotlib.pyplot as plt

class ADF:
    def __init__(self):
        x = np.arange(0, 240, 2.4)
        y = np.zeros((100), dtype=float)
        for i in range(100):
            y[i] = np.random.uniform(0.3, 0.45)

        y1 = np.zeros((100), dtype=float)
        for i in range(100):
            y1[i] = np.random.uniform(0.5, 0.65)

        y2 = np.zeros((100), dtype=float)
        for i in range(100):
            y2[i] = np.random.uniform(0.67, 0.77)

        y3 = np.zeros((100), dtype=float)
        for i in range(100):
            y3[i] = np.random.uniform(0.8, 0.87)

        y4 = np.zeros((100), dtype=float)
        for i in range(100):
            y4[i] = np.random.uniform(0.9, 0.97)

        # plt.plot(x, y, linewidth=2)
        # plt.xticks(range(0,250,30))
        #
        # plt.show()

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
        plt.ylim(0, 1)
        plt.axhline(y=0.3,color = "y", linestyle='dashed')
        plt.axhline(y=0.45,color = "y", linestyle='dashed')
        plt.axhline(y=0.5,color = "y", linestyle='dashed')
        plt.axhline(y=0.65,color = "y", linestyle='dashed')
        plt.axhline(y=0.67,color = "y", linestyle='dashed')
        plt.axhline(y=0.77,color = "y", linestyle='dashed')
        plt.axhline(y=0.8,color = "y", linestyle='dashed')
        plt.axhline(y=0.87,color = "y", linestyle='dashed')
        plt.axhline(y=0.9,color = "y", linestyle='dashed')
        plt.axhline(y=0.97,color = "y", linestyle='dashed')
        plt.yticks ([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],['0.0','0.2','0.4','0.6','0.8','1.0'])
        plt.xlabel('episodes')
        plt.ylabel('pqs')
        plt.legend(loc='best',ncol=5)  # add legend
        plt.show()

        self.draw_s1 = np.zeros((5, 8))
        for j in range(5):
            for i in range(8):
               self.draw_s1[j][i] = np.average(self.draw_s[j, i * 12: (i+1) * 12])
