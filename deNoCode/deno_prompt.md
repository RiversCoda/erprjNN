# 请生成一个神经网络降噪工程，具体要求如下：
## 环境：windows, pytorch, 4060ti(要求代码支持混合精度提速，和num_workers加速)
## 训练数据集：
噪音： `addNoise_data\20241113\noise\*\scg\*.mat`
安静label： `addNoise_data\20241113\raw\*\scg\*.mat`
对于每个安静数据，都有多条对应的噪音数据，例如：
``` 安静数据和噪音数据对应关系
addNoise_data\20241113\raw\hqw\scg\10-12-12-39-22_slice0.mat
——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise0.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise1.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise2.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise3.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise4.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise5.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise6.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise7.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise8.mat addNoise_data/20241113/noise/hqw/scg/10-12-12-39-22_slice0_noise9.mat
```
比如一条安静数据有10条噪音数据，则将10对数据送入网络训练，每次训练的输入这条安静数据和一条噪音数据，输出是去噪后的数据。
数据存储格式为.mat文件的accresult变量，形状为(4, 2000)。只使用第二条通道的数据进行训练和预测。（即0-3索引中的1索引）

## 测试数据集：
噪音： `addNoise_data\test\noise\*\scg\*.mat`
安静label： `addNoise_data\test\raw\*\scg\*.mat`
其他信息和训练数据集一致

## 程序要求：
### dn_dataloader.py
- 支持所有程序所需的数据读取

### dn_train.py
- 支持选择gan和transformer两种网络结构进行训练，支持混合精度训练，支持num_workers加速（gan的生成器输入是噪音数据，输出是去噪后的数据，判别器输入是去噪后的数据和安静数据；生成器用一个transformer，判别器用多层卷积神经网络）
- 要求尽量解耦合，方便后续扩展
- 每5轮保存一次模型（保存轮数参数和其他超参数写在程序前面，方便修改）
- 保存模型名字要求带有当前模型名称、超参数、轮数、损失值等信息
- 使用进度条tqdm显示训练进度

### dn_test.py
- 选择模型进行测试

### dn_model.py
- 支持gan和transformer(默认8层4头)两种网络结构

### dn_utils.py
- 用于存放一些通用的函数

### 额外要求
- 要求代码具有较高的模块性
