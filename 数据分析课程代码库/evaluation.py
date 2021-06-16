# -*- coding: utf-8 -*-

'''
#交叉验证模型
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

digits=datasets.load_digits()
features=digits.data
target=digits.target
standardizer=StandardScaler()
logit=LogisticRegression()

pipline=make_pipeline(standardizer,logit)
kf=KFold(n_splits=10,shuffle=True,random_state=1)
cv_result=cross_val_score(pipline,features,target,cv=kf,scoring='accuracy',n_jobs=-1)

print(cv_result)
print(cv_result.mean())
'''

#可视化不同训练集规模结果
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve

digits=load_digits()
features,target=digits.data,digits.target

train_sizes,train_scores,test_scores=learning_curve(
RandomForestClassifier(),features,target,cv=5,scoring='accuracy',
n_jobs=-1,train_sizes=np.linspace(0.01,1.0,50))

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)


plt.plot(train_sizes,train_mean,'--',color='#111111',label='Training score')
plt.plot(train_sizes,test_mean,color='#111111',label='Cross-validation score')

plt.fill_between(train_sizes,train_mean-train_std,
train_mean+train_std,color='#DDDDDD')
plt.fill_between(train_sizes,test_mean-test_std,
test_mean+test_std,color='#DDDDDD')

plt.title('Learning Curve')
plt.xlabel('Training set size')
plt.ylabel('Accuracy score')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


#生成模型评估报告
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris=datasets.load_iris()
features=iris.data
target=iris.target

class_names=iris.target_names
print(class_names)

features_train,features_test,target_train,target_test=train_test_split(
features,target,random_state=1)
classifier=LogisticRegression()

model=classifier.fit(features_train,target_train)
target_predict=model.predict(features_test)

print(classification_report(target_test,target_predict,target_names=class_names))
