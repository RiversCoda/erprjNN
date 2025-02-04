# 请生成一个神经网络降噪工程，具体要求如下：
## 环境：windows, pytorch, 4060ti（支持cuda）(要求代码支持混合精度提速，和num_workers加速)
## 训练数据集：
噪音路径： `addNoise_data\20241113\noise\*\scg\*.mat`
安静label路径： `addNoise_data\20241113\raw\*\scg\*.mat`
对于每个安静数据，都有多条对应的噪音数据，例如：
``` 安静数据和噪音数据对应关系
addNoise_data\20241113\raw\hqw\scg\10-12-12-39-22_slice0.mat
——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise0.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise1.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise2.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise3.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise4.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise5.mat addNoise_data\20241113\noise\hqw\scg\10-12-QD12-39-22_slice0_noise6.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise7.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise8.mat addNoise_data\20241113\noise\hqw\scg\10-12-12-39-22_slice0_noise9.mat
```
比如一条安静数据有10条噪音数据，则将10对数据送入网络训练，每次训练的输入这条安静数据和一条噪音数据，输出是去噪后的数据。
数据存储格式为.mat文件的accresult变量，形状为(4, 2000)。只使用第二条通道的数据（1*2000）进行训练和预测。（即0-3索引中的1索引）

## 测试数据集路径：
噪音路径： `addNoise_data\test\noise\*\scg\*.mat`
安静label路径： `addNoise_data\test\raw\*\scg\*.mat`
其他信息和训练数据集一致

## 任务：
`1*2000`输入（原始采样率为500hz），`1*2000`输出的有监督学习任务
使用MSE作为默认损失函数

## 模型结构
模型支持结合多种线路进行神经网络处理，具体种类如下:
- baseline: 一个4层4头的transformer网络,直接输入`1*2000`输出`1*2000`的有监督学习
- 行注意力：输入为1x2000的时间序列，通过STFT（参数为nperseg =  64, noverlap = 32, nfft = 512）转化为幅值或幅值+相位或实部+虚部2D频谱图（三种类型通过传参和if进行选择），仅保留0-200Hz的频率部分，进行一次norm。将频谱图按行拆分，每一行作为一个token输入到Transformer网络（需要能处理三种情况的格式编码为token，输入transformer）。Transformer通过行注意力机制处理这些token，并输出一个与输入大小相同的1x2000的特征表示。若输出尺寸不符合预期（如未达到1x2000），则可通过反卷积、上采样或全连接层等方式进一步调整输出尺寸以匹配目标尺寸。
- 列注意力：和行注意力逻辑相似，不同的是将频谱图按列拆分，每一列作为一个token输入到Transformer网络，Transformer通过列注意力机制处理这些token，并输出一个与输入大小相同的1x2000的特征表示。若输出尺寸不符合预期（如未达到1x2000），则可通过反卷积、上采样或全连接层等方式进一步调整输出尺寸以匹配目标尺寸。
- 组合注意力：结合行注意力、列注意力的逻辑，通过通道卷积，结合两者的输出获得最终的输出
- 多中组合：集合baseline、行注意力、列注意力的逻辑，结合多种输出获得最终的输出

## 程序要求：
### dn_dataloader.py
- 支持所有程序所需的数据读取
- 将数据读取后首先进行标准化（支持两种标准化进行选择，根据传入的参数，选择0到1的标准化或指定均值方差的标准化）
- 根据传入的参数选择是否进行低通滤波

### dn_train.py
- 使用Adam优化器
- transformer两种网络结构进行训练，支持混合精度训练，支持num_workers加速（gan的生成器输入是噪音数据，输出是去噪后的数据
- 每5轮保存一次模型到`save_models\`文件夹（需要创建）（保存轮数参数和其他超参数写在程序前面，方便修改）
- 保存模型名字要求带有当前模型名称、超参数、轮数、损失值等信息
- 使用进度条tqdm显示训练进度
- 训练开始后生成一条params_time.txt(time是当前日期时间格式为MMDDhhmmss)文件保存在trainDetails文件夹（若不存在需要创建），记录当次训练全部的超参数、transformer头数、数据预处理方式、是否进行了滤波、训练集路径、训练集数据条数等各种运行信息

### dn_test.py
- 选择模型进行测试
- 测试开始后生成一条test_time.txt文件(time是当前日期时间格式为MMDDhhmmss)保存在testDetails文件夹

### dn_model.py
- 支持模型设计中提到的各种网络结构

### dn_utils.py
- 用于存放一些通用的函数，例如标准化、滤波、stft等

### 额外要求
- 要求代码具有较高的模块性
- 保证代码能够支持快速切换不同模型结构和数据预处理策略，确保易于扩展和维护。
