#coding:gbk,
#异常检测
from sklearn.svm import OneClassSVM  # 导入OneClassSVM
import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入Matplotlib
from mpl_toolkits.mplot3d import Axes3D  # 导入3D样式库

# 数据准备
raw_data = np.loadtxt('outlier.txt', delimiter=' ')  # 读取数据
print(raw_data.shape)
train_set = raw_data[:900, :]  # 训练集
test_set = raw_data[900:, :]  # 测试集

# 异常数据检测
model_onecalsssvm = OneClassSVM(nu=0.3, kernel="rbf")  # 创建异常检测算法模型对象
model_onecalsssvm.fit(train_set)  # 训练模型
pre_test_outliers = model_onecalsssvm.predict(test_set)  # 异常检测,1标识正常数据，-1标识异常数据
print(pre_test_outliers.shape)

# 异常结果统计
toal_test_data = np.hstack((test_set, pre_test_outliers.reshape(test_set.shape[0], 1)))  # 将测试集和检测结果合并
#vstack()#在竖直方向拼接数组
normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # 获得异常检测结果中正常数据集
outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # 获得异常检测结果中异常数据
n_test_outliers = outlier_test_data.shape[0]  # 获得异常的结果数量
total_count_test = toal_test_data.shape[0]  # 获得测试集样本量
print('outliers: {0}/{1}'.format(n_test_outliers, total_count_test))  # 输出异常的结果数量
print('{:*^60}'.format(' all result data (limit 5) '))  # 打印标题
print(toal_test_data[:5])  # 打印输出前5条合并后的数据集

# 异常检测结果展示
plt.style.use('ggplot') # 使用ggplot样式库
fig = plt.figure()  # 创建画布对象
ax = Axes3D(fig)  # 将画布转换为3D类型
s1 = ax.scatter(normal_test_data[:, 0], normal_test_data[:, 1], normal_test_data[:, 2], s=100, edgecolors='k', c='g',
                marker='o')  # 画出正常样本点
s2 = ax.scatter(outlier_test_data[:, 0], outlier_test_data[:, 1], outlier_test_data[:, 2], s=100, edgecolors='k', c='r',
                marker='o')  # 画出异常样本点
ax.w_xaxis.set_ticklabels([])  # 隐藏x轴标签，只保留刻度线
ax.w_yaxis.set_ticklabels([])  # 隐藏y轴标签，只保留刻度线
ax.w_zaxis.set_ticklabels([])  # 隐藏z轴标签，只保留刻度线
ax.legend([s1, s2], ['normal points', 'outliers'], loc=0)  # 设置两类样本点的图例
plt.title('novelty detection')  # 设置图像标题
plt.show()  # 展示图像
