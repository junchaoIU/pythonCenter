# -------------------------------------------------------------------------------
# Description:  
# Reference:
# Name:   结构化数据建模实验
# Author: wujun
# Date:   2021/6/12
# -------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models,layers

# titanic数据集的目标是根据乘客信息预测他们在Titanic号撞击冰山沉没后能否生存
# 结构化数据一般会使用Pandas中的DataFrame进行预处理
dftrain_raw = pd.read_csv('./titanic/train.csv')
dftest_raw = pd.read_csv('./titanic/test.csv')
print(dftrain_raw.head(10))

## 字段及处理说明：
# * Survived:0代表死亡，1代表存活【y标签】
# * Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# * Name:乘客姓名 【舍去】
# * Sex:乘客性别 【转换成bool特征】
# * Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# * SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# * Parch:乘客父母/孩子的个数(整数值)【数值特征】
# * Ticket:票号(字符串)【舍去】
# * Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# * Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# * Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】

# 利用Pandas的数据可视化功能我们可以简单地进行探索性数据分析EDA（Exploratory Data Analysis）。
# label分布情况
ax = dftrain_raw['Survived'].value_counts().plot(kind = 'bar',
     figsize = (12,8),fontsize=15,rot = 0)
ax.set_ylabel('Counts',fontsize = 15)
ax.set_xlabel('Survived',fontsize = 15)
plt.show()

# 年龄分布情况
ax = dftrain_raw['Age'].plot(kind = 'hist',bins = 20,color= 'purple',
                    figsize = (12,8),fontsize=15)
ax.set_ylabel('Frequency',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()

# 年龄和label的相关性
ax = dftrain_raw.query('Survived == 0')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
dftrain_raw.query('Survived == 1')['Age'].plot(kind = 'density',
                      figsize = (12,8),fontsize=15)
ax.legend(['Survived==0','Survived==1'],fontsize = 12)
ax.set_ylabel('Density',fontsize = 15)
ax.set_xlabel('Age',fontsize = 15)
plt.show()

# 数据预处理
def preprocessing(dfdata):

    dfresult= pd.DataFrame()

    #Pclass 乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' +str(x) for x in dfPclass.columns ]
    dfresult = pd.concat([dfresult,dfPclass],axis = 1)

    #Sex 乘客性别 【转换成bool特征】
    dfSex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult,dfSex],axis = 1)

    #Age
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    #SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    #Carbin
    dfresult['Cabin_null'] =  pd.isna(dfdata['Cabin']).astype('int32')

    #Embarked
    dfEmbarked = pd.get_dummies(dfdata['Embarked'],dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult,dfEmbarked],axis = 1)

    return(dfresult)

x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

print("x_train.shape =", x_train.shape )
print("x_test.shape =", x_test.shape )

# 定义模型
# 使用Keras接口有以下3种方式构建模型：使用Sequential按层顺序构建模型，使用函数式API构建任意结构模型，继承Model基类构建自定义模型。
# 此处选择使用最简单的Sequential，按层顺序模型。
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(layers.Dense(20,activation = 'relu',input_shape=(15,)))
model.add(layers.Dense(10,activation = 'relu' ))
model.add(layers.Dense(1,activation = 'sigmoid' ))

model.summary()

# 训练模型
# 训练模型通常有3种方法，内置fit方法，内置train_on_batch方法，以及自定义训练循环。此处我们选择最常用也最简单的内置fit方法。
# 二分类问题选择二元交叉熵损失函数
model.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['AUC'])

history = model.fit(x_train,y_train,
                    batch_size= 64,
                    epochs= 30,
                    validation_split=0.2 #分割一部分训练数据用于验证
                   )

# 评估模型
# 我们首先评估一下模型在训练集和验证集上的效果
def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history,"loss")

plot_metric(history,"AUC")

# 我们再看一下模型在测试集上的效果.
model.evaluate(x = x_test,y = y_test)

# 使用模型
#预测概率
model.predict(x_test[0:10])
#model(tf.constant(x_test[0:10].values,dtype = tf.float32)) #等价写法

# 预测类别
# model.predict_classes(x_test[0:10])

# 保存模型
#　可以使用Keras方式保存模型，也可以使用TensorFlow原生方式保存。
# 前者仅仅适合使用Python环境恢复模型，后者则可以跨平台进行模型部署。
# 推荐使用后一种方式进行保存。
# 1，Keras方式保存
# 保存模型结构及权重
model.save('keras_model.h5')
del model  #删除现有模型

# identical to the previous one
model = models.load_model('keras_model.h5')
model.evaluate(x_test,y_test)

# 保存模型结构
json_str = model.to_json()
# 恢复模型结构
model_json = models.model_from_json(json_str)

#保存模型权重
model.save_weights('keras_model_weight.h5')

# 恢复模型结构
model_json = models.model_from_json(json_str)
model_json.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
# 加载权重
model_json.load_weights('keras_model_weight.h5')
model_json.evaluate(x_test,y_test)

# 2，TensorFlow原生方式保存
# 保存权重，该方式仅仅保存权重张量
model.save_weights('./tf_model_weights.ckpt',save_format = "tf")
# 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署
model.save('tf_model_savedmodel', save_format="tf")
print('export saved model.')

model_loaded = tf.keras.models.load_model('tf_model_savedmodel')
model_loaded.evaluate(x_test,y_test)