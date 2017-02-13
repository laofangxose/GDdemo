
之所以在这个公众号上写学习笔记，是因为最近一直在上一些机器学习相关的网络课程，发现有很多概念比较抽象，所以把学习过程中的一些重点问题整理记录下来，方便自己今后的学习，同时也欢迎各位读者共同探讨，一起提高。

在学习机器学习的过程中，肯定离不开的优化算法就是gradient descent梯度下降，这是一个原理和数学上都比较好理解的一个算法。gradient是梯度，意义是一个变量值上升最快的方向，即梯度的相反数是下降最快的方向；那么gradient descent，就可以理解为，将我们需要优化（求最小值）的变量，沿着它值减小最快的方向前进，通过不断的迭代，最终达到最低点。

对于通常的优化问题，我们都会有一个cost function，用$J(\theta)$表示，$\theta$代表样本的特征feature，如果样本有多个features，则$\theta$表示为矩阵。这个cost function，我们要求它在点$\theta$处可微，这样我们就可以通过gradient descent来求它的最小值，于是我们随机选取一个起点$\theta_0$，然后求$J(\theta)$在$\theta_0$处的梯度，再沿着梯度前进一定的距离，就完成了一次迭代。这个前进的距离又叫learning rate $\alpha$，是一个由我们控制的量。每一次迭代的过程可以用下式表达：

<img src="http://chart.googleapis.com/chart?cht=tx&chl= \theta_{n+1}=\theta_n-\alpha\nabla J(\theta_n)" style="border:none;">


注意式子中的负号，这是表示逆着梯度的方向，但是这里是负号不代表实际的表达式中一定会带有负号，可能因为某些计算上的原因负号被消掉了，但是原理上一定要记得这个负号的存在。

下图类似于地理上等高线的画法，每一个圆圈代表cost function相等的点，而红色的箭头代表每一次gradient descent下降的方向，可以看到随着不断的迭代，我们在逐渐靠近最低点。
![Gradient_descent](https://upload.wikimedia.org/wikipedia/commons/7/79/Gradient_descent.png?download)

如果learning rate选择的太小，我们靠近最低点的速度就会放缓，如果learning rate选择的太大，则可能过犹不及，在最低点附近震荡，甚至逐渐远离最低点（不收敛）。

如果cost function是convex的，那么我们最终会停在最低点附近；如果不是，那么可能存在多个极小值，而不同的出发点可能会收敛不同的局部极小值，这也是可能存在的问题。

在上面的算法里，每次迭代，都要把所有的样本拿出来计算梯度，然而这样做在样本数量很大的时候，会浪费大量的计算资源，一个比较好的替代方法是将所有N个样本分成许多小份，每份M个样本，在每次迭代的时候随机选择一小份进行计算，这样的话牺牲了一些收敛的速度，但是却简化了计算，提高了效率。

以最简单的直线线性回归Linear Regression为例，我们需要找到一条直线使得所有数据点到这条直线距离的平方和最小（最小二乘）。下面放一个简单的Gradient descent的python实例，感兴趣的朋友可以跑一下看看。


```Python
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
```

点击“阅读原文”下载代码
