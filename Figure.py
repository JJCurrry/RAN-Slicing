from pylab import * #作图用
from matplotlib import rcParams
config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config) #图例显示汉字宋体和新罗马


# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8,6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(1,2,1)

X = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Y1 = [13, 24, 39, 46.5, 56.5, 62, 65.3, 70.4, 71.2, 79]
# Y2 = [13, 23.7, 39.4, 47, 59, 67, 69, 74.3, 81, 96]
Y1 = [3.63, 3.71, 3.75, 3.82, 3.96, 4.33, 4.77, 5.16, 5.7, 5.73]
Y2 = [3.6, 3.73, 3.735, 3.8, 3.79, 4, 4.56, 4.76, 5.03, 5.1]

# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, Y1, color="green", marker='s', linewidth=1.5, linestyle="-", label="基于Q-Learning算法")

# # 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
# plot(X, Y2, color="blue", marker='*', linewidth=1.5, linestyle="-", label="基于DQN算法")

# 设置横轴的上下限
xlim(5,50)
xlabel("uRLLC用户数量")

# 设置纵轴的上下限
ylim(3.5,6)
ylabel("平均时延 单位：ms")


subplot(1,2,2)

X = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
# Y1 = [13, 24, 39, 46.5, 56.5, 62, 65.3, 70.4, 71.2, 79]
# Y2 = [13, 23.7, 39.4, 47, 59, 67, 69, 74.3, 81, 96]
Y1 = [3.63, 3.71, 3.75, 3.82, 3.96, 4.33, 4.77, 5.16, 5.7, 5.73]
Y2 = [3.6, 3.73, 3.735, 3.8, 3.79, 4, 4.56, 4.76, 5.03, 5.1]
# # 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
# plot(X, Y2, color="green", marker='s', linewidth=1.5, linestyle="-", label="基于Q-Learning算法")

# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, Y2, color="blue", marker='*', linewidth=1.5, linestyle="-", label="基于DQN算法")

# 设置横轴的上下限
xlim(5,50)
xlabel("uRLLC用户数量")

# 设置纵轴的上下限
ylim(3.5,6)
ylabel("平均时延 单位：ms")


# 以分辨率 1080 来保存图片
legend(loc=0)
savefig("p.png",dpi=1080)
# 在屏幕上显示
show()