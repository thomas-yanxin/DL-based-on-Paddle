# PaddlePaddle实现波士顿房价预测


```python
#加载飞桨、Numpy 和相关库
import paddle
import paddle.fluid as fluid               # 飞桨主库
import paddle.fluid.dygraph as dygraph     # 动态图类库
from paddle.fluid.dygraph import Linear
import numpy as np
import random
import os
```

### 数据预处理

&emsp;&emsp;数据预处理主要包含五个部分：数据导入、数据形状变换、数据集划分、数据归一化处理、封装 load_data 函数。


```python
# 数据预处理

def load_data():
    # 从文件读取数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=" ")     # 从文本或二进制文件中构造一个数组

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    # 统计字段个数
    feature_num = len(feature_name)

    # 将原始数据进行reshape， 变成 [N, 14]的形状
    # N = data.shape[0]//feature_num
    data = data.reshape([data.shape[0]//feature_num, feature_num])

    # 将原始数据及拆分为训练集和测试集（8：2）
    ratio = 0.8
    offset = int(data.shape[0]*ratio)
    training_data = data[:offset]
    
    # 计算训练集的max, min, mean
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的化分
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data



```


```python
# 查看数据
training_data , test_data = load_data()
print(training_data,test_data)

print("~"*20)

# 查看第一个训练样本数据
x = training_data[:, :-1]
y = training_data[:, -1:]
print(x[0])     #前13个影响因素
print(y[0])     #第14个房价中位数
```

### 搭建神经网络

&emsp;&emsp;线性回归模型采用线性激活函数( linear activation )的全连接层 ( fully-connected layer, fc_layer )，因此在飞桨中利用全连接层模型构造线性回归，这样，一个全连接层就可以看作一个简单的神经网络。  

&emsp;&emsp;搭建神经网络类似于使用积木搭建宝塔。在飞桨中，网络层（layer）是积木，而神经网络是要搭建的宝塔。我们使用不同的layer进行组合，来搭建神经网络。**飞桨建议通过创建Python类的方式完成模型网络的定义，即__init__函数和forward函数。**  

&emsp;&emsp;**forward函数是框架指定实现前向计算逻辑的函数，程序在调用模型实例时会自动执行forward方法。在forward函数中使用的网络层需要在__init__函数中声明。**  

&emsp;&emsp;**定义init函数**：在类的初始化函数中声明每一层网络的实现函数。  

&emsp;&emsp;**定义forward函数**：构建神经网络结构，实现前向计算过程，并返回预测结果。



```python
# 配置网络结构
class Regressor(fluid.dygraph.Layer):
    def __init__(self,name_scope):
        super(Regressor,self).__init__(name_scope)
        name_scope = self.full_name()
        # 定义一层全连接层，输出维度是1， 激活函数为None, 即不使用激活函数
        self.fc = Linear(input_dim = 13, output_dim = 1,act = None)

    # 网络的前向计算函数
    def forward(self,input):
        x = self.fc(input)
        return x


```

### 训练配置
1. 指定运行训练的机器资源：以guard函数指定运行训练的机器资源，表明在with作用域下的程序均执行在本机的CPU资源上。 dygraph.guard 表示在with作用于下的程序会以动态图的模式执行（实时执行）。
2. 声明模型实例：声明定义好的回归模型Regressor实例，并将模型的状态设置为训练。
3. 加载训练和测试数据：使用load_data 函数加载训练数据和测试数据。
4. 设置优化算法和学习率：优化算法采用随机梯度下降SGD，学习率设置为0.01.



```python
# 初始化

with fluid.dygraph.guard():
    '''

        在paddlepaddle中，模型实例有两个状态：train()和eval()。
        训练时要执行正向计算和反向传播梯度两个过程，而预测只需要执行正向计算。
        另外，with fluid.dygraph.guard()创建了飞桨动态图的工作环境，在该环境中完成模型声明、数据转换及模型训练等。

    '''
    # 声明定义好的线性回归模型（Regressor）
    model = Regressor('Regressor')
    model.train()
    # 数据加载
    training_data, test_data = load_data()
    print(training_data[10:20])
    # 定义优化算法，SGD
    # 学习率设置为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01, parameter_list= model.parameters())

```

### 模型训练
&emsp;&emsp;模型训练过程采用**内层循环**和**外层循环嵌套**的方式。  

&emsp;&emsp;内层循环负责整个数据集的一次遍历，采用分批次(batch)方式。Batch的取值会影响模型训练效果：batch过大，会增大内存消耗和计算时间，且效果不会明显提升；batch过小，每个batch的样本数据将没有统计意义。  

&emsp;&emsp;**内循环四个步骤**：
1. **数据准备**：将一个批次的数据转变为np.array和内置格式。
2. **前向计算**：将一个批次的样本数据灌入网络中，计算输出结果。
3. **计算损失函数**：以前向计算结果和真实房价作为输入，通过损失函数square_error_cost计算出损失函数值（Loss）.
4. **反向传播**： 执行梯度反向传播backward函数，即从后到前逐层计算每一层的梯度，并根据设置的优化算法更新参数opt.minimize。  

&emsp;&emsp;外层循环定义遍历数据集的次数，通过参数EPOCH_NUM设置。









```python
# 定义训练过程
with dygraph.guard(fluid.CPUPlace()):
    EPOCH_NUM = 10     # 设置外层循环次数
    BATCH_SIZE = 10    # 设置batch大小
    
    # 定义外层循环
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机打乱
        np.random.shuffle(training_data)
        # 将训练数据进行拆分，每个batch包含10条数据
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0,len(training_data),BATCH_SIZE)]

        # 定义内层循环
        for iter_id, mini_batch in enumerate(mini_batches):
            # 获得当前批次训练数据
            x = np.array(mini_batch[:,:-1]).astype('float32')
            # 获得当前批次训练标签（真是房价）
            y = np.array(mini_batch[:,-1:]).astype('float32')
            # 将numpy数据转为飞桨动态图variable形式
            house_features = dygraph.to_variable(x)
            prices = dygraph.to_variable(y)

            # 正向计算
            predicts = model(house_features)

            # 计算损失
            loss = fluid.layers.square_error_cost(predicts, label=prices)
            avg_loss = fluid.layers.mean(loss)
            if iter_id % 20 == 0:
                print("epoch:{}, iter:{}, loss is:{}".format(epoch_id,iter_id,avg_loss.numpy()))
            
            #反向传播
            avg_loss.backward()
            # 最小化loss, 更新参数
            opt.minimize(avg_loss)
            # 清除梯度
            model.clear_gradients()

    fluid.save_dygraph(model.state_dict(),"LR_model")


```

### 保存并测试模型
&emsp;&emsp;首先我们将模型当前的参数数据model.state_dict()保存在文件中（通过参数指定保存的文件LR_model），以备预测或校验的程序调用。


```python
# 定义飞桨动态图工作环境
with fluid.dygraph.guard():
    # 保存模型参数，文件为LR_model
    fluid.save_dygraph(model.state_dict(),'LR_model')
    print("模型保存成功，模型参数保存在LR_model中")

```

    模型保存成功，模型参数保存在LR_model中


&emsp;&emsp;然后可以对模型进行测试，测试过程与在应用场景中使用模型的过程一致，主要分为如下三个步骤：  

（1）、配置模型预测的机器资源；  

（2）、将训练好的模型参数加载到模型。加载完毕后，需要将模型的状态调整为evaluation（校验）。  

（3）、将待预测的样本特征输入模型中，打印输出的预测结果。  

&emsp;&emsp;通过load_one_example函数从数据集中抽出一条样本作为测试样本。



```python
# 读取测试样本
def load_one_example(data_dir):
    f = open(data_dir, 'r')
    datas = f.readlines()
    # 选择倒数第十条数据用于测试
    tmp = datas[-10]
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]

    # 对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i] - avg_values[i]) / (max_values[i] - min_values[i])
    
    data = np.reshape(np.array(one_data[:-1]), [1,-1]).astype(np.float32)
    label = one_data[-1]
    return data, label

# 测试模型
with dygraph.guard():
    # 参数为保存模型参数的文件地址
    model_dict, _ = fluid.load_dygraph("LR_model")
    model.load_dict(model_dict)
    model.eval()

    # 参数为数据集的文件地址
    test_data, label = load_one_example('./work/housing.data')
    # 将数据转为动态图的variable格式
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)

    # 对结果进行反归一化处理
    results = results*(max_values[-1] - min_values[-1]) +avg_values[-1]
    print("Inference result is {}, the corresponding label is {}".format(results.numpy(),label))

```

    Inference result is [[14.326102]], the corresponding label is 19.7

