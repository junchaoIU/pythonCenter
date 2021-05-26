# -------------------------------------------------------------------------------
# Name:   work2
# Author: wujunchao
# Date:   2021/5/20
# -------------------------------------------------------------------------------

import pandas as pd
from IPython.display import display

print('='*20+"数据预处理"+'='*20)
# 打开文件
train = pd.read_excel('order1.xlsx',sheet_name='Sheet1')
test = pd.read_excel('order1.xlsx',sheet_name='Sheet2')

# 查看训练集和测试集前5行
print("训练集和测试集前5行:")
display(train.head(5),test.head(5))

# 查看训练集和测试集样本量及特征量
print('train dataset: {0}; predict dataset: {1}'.format(
    train.shape, test.shape))

# 查看数据中缺失值的量
print("查看数据中缺失值的量：")
display(
    train.isnull().sum(),
    test.isnull().sum()
)

# 删除有空值的行
train = train.dropna()

# 测试集的预测标签字段名修改为response，方便观察
test.rename(columns={'final_response':'response'},inplace=True)

# 对连续型特征进行分箱（'age', 'edu_ages', 'total_pageviews', 'work_hours'）
train["age"] = pd.cut(train["age"], 5, labels=[1, 2, 3, 4, 5]).astype("int")  # 分箱
test["age"] = pd.cut(test["age"], 5, labels=[1, 2, 3, 4, 5]).astype("int")  # 分箱
train["total_pageviews"] = pd.cut(train["total_pageviews"], 5, labels=[5, 4, 3, 2, 1]).astype("int")  # 分箱
test["total_pageviews"] = pd.cut( test["total_pageviews"], 5, labels=[5, 4, 3, 2, 1]).astype("int")  # 分箱
train["edu_ages"] = pd.cut(train["edu_ages"], 4, labels=[4, 3, 2, 1]).astype("int")  # 分箱
test["edu_ages"] = pd.cut( test["edu_ages"], 4, labels=[4, 3, 2, 1]).astype("int")  # 分箱
train["work_hours"] = pd.cut(train["work_hours"], 4, labels=[4, 3, 2, 1]).astype("int")  # 分箱
test["work_hours"] = pd.cut( test["work_hours"], 4, labels=[4, 3, 2, 1]).astype("int")  # 分箱

# 丢弃无用的特征（'blue_money', 'red_money'）
train.drop(['blue_money', 'red_money'], axis=1, inplace=True)
test.drop(['blue_money', 'red_money'], axis=1, inplace=True)

# 去除异常值，离群点
def abnormal_detect(df):
    # 通过Z-Score方法判断异常值
    df_zscore = df.copy()  # 复制一个用来存储Z-score得分的数据框
    cols = df.columns  # 获得数据框的列名

    for col in cols:  # 循环读取每列
        df_col = df[col]  # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()  # 计算每列的Z-score得分（（每个元素-该列平均值）/该列标准差）
        df_zscore[col] = z_score.abs() > 2.2  # 判断Z-score得分的绝对值是否大于2.2，如果是则是True，否则为False
    # print(df_zscore)  # 打印输出

    # 删除异常值所在的行
    for col in cols:  # 循环读取每列删除
        df_drop = df[df_zscore[col] == False]
    return df_drop
# 去异常值
abnormal_detect(train)
abnormal_detect(test)

# 查看训练集和测试集前5行
print("再查看训练集和测试集：")
display(train.head(5),test.head(5))

# 切分输入训练集和测试集的特征X和预测y
y_train = train["response"].values
X_train = train.drop(['response'], axis=1).values
y_test = test["response"].values
X_test = test.drop(['response'], axis=1).values

# 查看数据
print('train dataset: {0}; test dataset: {1}'.format(
    X_train.shape, X_test.shape))



print('='*20+"KNN算法实现"+'='*20)

# KNN算法实现类
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn

training_accuracy = []
test_accuracy = []

# n_neighbors取值从1到19
neighbors_settings = range(1, 19)
for n_neighbors in neighbors_settings:
    # 构建模型
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # 记录训练集精度
    training_accuracy.append(knn.score(X_train, y_train))
    # 记录测试集精度
    test_accuracy.append(knn.score(X_test, y_test))

    #画图
plt.figure()
plt.title("KNN_score Graph")
plt.plot(neighbors_settings, training_accuracy, 'o-', color="y", label="training accuracy")   # training accuracy类别标签
plt.plot(neighbors_settings, test_accuracy, 'o-', color="g", label="test accuracy")    # test accuracy类别标签
plt.ylabel("Accuracy")    # y坐标
plt.xlabel("n_neighbors")     # x坐标
plt.grid()
plt.legend(loc='best')
plt.show()

# 得到最优参为4构建模型
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)  # 拟合模型
# 对测试集进行预测
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
# 打印结果
print('KNN算法结果：train score: {0}; test score: {1}'.format(train_score, test_score))
# KNN算法结果：train score: 0.8774851826843724; test score: 0.8947368421052632




print('='*20+"决策树算法实现"+'='*20)

# 决策树
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('train score: {0}; test score: {1}'.format(train_score, test_score))
# train score: 1.0; test score: 0.5789473684210527, 明显过拟合

from sklearn.tree import export_graphviz

with open("order1.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)
# 1. 在电脑上安装 graphviz；2. 运行 `dot -Tpng order1.dot -o order1.png` ；3. 在当前目录查看生成的决策树 order1.png

import numpy as np
import matplotlib.pyplot as plt

# 参数选择 max_depth
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)

depths = range(2, 15)
scores = [cv_score(d) for d in depths]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]

best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = depths[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))
# best param: 6; best score: 0.8421052631578947

plt.figure(figsize=(10, 6), dpi=144)
plt.grid()
plt.xlabel('max depth of decision tree')
plt.ylabel('score')
plt.plot(depths, cv_scores, '.g-', label='cross-validation score')
plt.plot(depths, tr_scores, '.r--', label='training score')
plt.legend()
plt.show()

# 训练模型，并计算评分
def cv_score(val):
    clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
    clf.fit(X_train, y_train)
    tr_score = clf.score(X_train, y_train)
    cv_score = clf.score(X_test, y_test)
    return (tr_score, cv_score)
# 指定参数范围，分别训练模型，并计算评分
values = np.linspace(0, 0.005, 50)
scores = [cv_score(v) for v in values]
tr_scores = [s[0] for s in scores]
cv_scores = [s[1] for s in scores]
# 找出评分最高的模型参数
best_score_index = np.argmax(cv_scores)
best_score = cv_scores[best_score_index]
best_param = values[best_score_index]
print('best param: {0}; best score: {1}'.format(best_param, best_score))
# best param: 0.0006122448979591836; best score: 0.8947368421052632

# 画出模型参数与模型评分的关系
plt.figure(figsize=(10, 6), dpi=144)
plt.grid()
plt.xlabel('threshold of entropy')
plt.ylabel('score')
plt.plot(values, cv_scores, '.g-', label='cross-validation score')
plt.plot(values, tr_scores, '.r--', label='training score')
plt.legend()
# 没有明显过拟合

# 学习曲线
def plot_curve(train_sizes, cv_results, xlabel):
    train_scores_mean = cv_results['mean_train_score']
    train_scores_std = cv_results['std_train_score']
    test_scores_mean = cv_results['mean_test_score']
    test_scores_std = cv_results['std_test_score']
    plt.figure(figsize=(10, 6), dpi=144)
    plt.title('parameters turning')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel('score')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, '.--', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, '.-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()  # 展示图像

# 网格搜索
from sklearn.model_selection import GridSearchCV
thresholds = np.linspace(0, 0.005, 50)
# Set the parameters by cross-validation
param_grid = {'min_impurity_decrease': thresholds}

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, return_train_score=True)
clf.fit(X_train, y_train)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,clf.best_score_))

plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')
# best param: {'min_impurity_decrease': 0.00040816326530612246}
# best score: 0.8670818015855153

# 网格搜索
from sklearn.model_selection import GridSearchCV

entropy_thresholds = np.linspace(0, 0.01, 50)
gini_thresholds = np.linspace(0, 0.005, 50)

# Set the parameters by cross-validation
param_grid = [{'criterion': ['entropy'],
               'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'],
               'min_impurity_decrease': gini_thresholds},
              {'max_depth': range(2, 10)},
              {'min_samples_split': range(2, 30, 2)}]

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5, return_train_score=True)
clf.fit(X_train, y_train)
print("best param: {0}\nbest score: {1}".format(clf.best_params_,
                                                clf.best_score_))
# 得到best param: {'criterion': 'entropy', 'min_impurity_decrease': 0.0006122448979591836}
# best score: 0.8672568584790057

# 最优参代入构建决策树模型
clf = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0006122448979591836)
clf.fit(X_train, y_train) # 拟合
train_score = clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)
print('决策树算法结果：train score: {0}; test score: {1}'.format(train_score, test_score))
# 决策树算法结果：train score: 0.8709830694975867; test score: 0.7894736842105263


