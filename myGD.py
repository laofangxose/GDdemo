import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    # y = mx + b
    def LeastMeanSquare(self,N,m,b,points):
        error = 0
        for i in range(N):
            error += (points[i, 1] - (m * points[i, 0] + b)) ** 2 # 求误差平方和
        return error/float(N)

    def GradientDescent(self,N,m,b,points,learningRate):
        b_gradient = 0
        m_gradient = 0
        for i in range(N):
            x = points[i, 0]
            y = points[i, 1]
            b_gradient += -(2/float(N)) * (y - ((m * x) + b)) # 计算b的梯度
            m_gradient += -(2/float(N)) * x * (y - ((m * x) + b)) # 计算m的梯度
            new_b = b - (learningRate * b_gradient) # 计算下降后的b
            new_m = m - (learningRate * m_gradient) # 计算下降后的m
        return [new_b, new_m]
    
    def run(self):
        learningRate = 0.02 #学习速率
        b = 0 # 初始截距
        m = 0 # 初始斜率
        NumIterations = 100 # 迭代次数

        data = np.array([[1, 2],[2, 3.1],[3, 6],[2.5,3.76],[0.6,1.4],[3.3,5.4]])
        N = len(data)
        
        fig,ax=plt.subplots()

        
        x = np.linspace(-1,5,100)
        for i in range(NumIterations):
            b, m = self.GradientDescent(N, m, b, data, learningRate)  # 迭代计算斜率m和截距b
            # 以下为画图
            ax.cla()
            for j in range(N):
                ax.scatter(data[j,0],data[j,1]) 
            ax.plot(x,(m*x+b))
            ax.set_xlim(-1,5)
            ax.set_ylim(-1,10)
            plt.pause(0.01)

        plt.show()
        

lr = LinearRegression()
lr.run()