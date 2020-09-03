import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers

# 载入结构化数据
dftrain_raw = pd.read_csv('../data/titanic/train.csv')
dftest_raw = pd.read_csv('../data/titanic/test.csv')
# print(dftrain_raw.head(10))

# 存活分布情况
ax = dftrain_raw['Survived'].value_counts().plot(kind='bar', figsize=(12, 8),
                                                 fontsize=15, rot=0)

ax.set_ylabel('Counts', fontsize=15)
ax.set_xlabel('Survived', fontsize=15)

plt.figure()

# 年龄与存活的相关性情况
ax = dftrain_raw.query('Survived == 1')['Age'].plot(kind='density',
                                                    figsize=(12, 8), fontsize=15)
dftrain_raw.query('Survived == 0')['Age'].plot(kind='density',
                                               figsize=(12, 8), fontsize=15)

ax.legend(['Survived==0', 'Survived==1'], fontsize=12)
ax.set_ylabel('Density', fontsize=15)
ax.set_xlabel('Age', fontsize=15)


# 结构化数据预处理
def preprocessing(dfdata):
    # 利用panda的数据结构进行预处理
    dfresult = pd.DataFrame()

    # Pclass one-hot for 3 dimensions
    dfPclass = pd.get_dummies(dfdata['Pclass'])
    dfPclass.columns = ['Pclass_' + str(x) for x in dfPclass.columns]
    dfresult = pd.concat([dfresult, dfPclass], axis=1)

    # Sex
    dfsex = pd.get_dummies(dfdata['Sex'])
    dfresult = pd.concat([dfresult, dfsex], axis=1)

    # Age:空缺补零
    dfresult['Age'] = dfdata['Age'].fillna(0)
    dfresult['Age_null'] = pd.isna(dfdata['Age']).astype('int32')

    # SibSp,Parch,Fare
    dfresult['SibSp'] = dfdata['SibSp']
    dfresult['Parch'] = dfdata['Parch']
    dfresult['Fare'] = dfdata['Fare']

    # Carbin 判断null便于排除其它字符串的干扰
    dfresult['Cabin_null'] = pd.isna(dfdata['Cabin']).astype('int32')

    # Embarked one-hot
    dfEmbarked = pd.get_dummies(dfdata['Embarked'], dummy_na=True)
    dfEmbarked.columns = ['Embarked_' + str(x) for x in dfEmbarked.columns]
    dfresult = pd.concat([dfresult, dfEmbarked], axis=1)

    return (dfresult)


x_train = preprocessing(dftrain_raw)
y_train = dftrain_raw['Survived'].values

x_test = preprocessing(dftest_raw)
y_test = dftest_raw['Survived'].values

# print("x_train.shape =", x_train.shape)
# print("x_test.shape =", x_test.shape)

# 模型定义
model = models.Sequential()
# 此处input_shape只需要声明特征维度即可
model.add(layers.Dense(20, activation='relu', input_shape=(15,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

# 二分类问题选择二元交叉熵
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2
)


def plot_metric(history, metric):
    plt.figure()
    train_metrics = history.history[metric]
    val_metrics = history.history['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(history, 'loss')
plot_metric(history, 'accuracy')
plt.show()

# 测试集评估
model.evaluate(x = x_test,y = y_test)

# save
# keras mode
model.save('./save_model/exp_structure.h5')

# tensorflow mode
# model.save_weights('tf_model_savemodel')
# model.save('./save_model/tf_model', save_format='tf')
