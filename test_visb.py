import torch 
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score  # 引入召回率和精确率
from torch.utils.data import DataLoader
from dataloader import HeartbeatDataset
from models import HeartbeatResNet
from sklearn.manifold import TSNE
import numpy as np

# 测试数据集目录
test_dir = 'test_data/device3/p4-test/'
# 已保存模型的目录
model_dir = 'save_models'
# 结果文件，用于存储模型评估结果
result_file = 'result.txt'

# 加载测试数据集
test_dataset = HeartbeatDataset(test_dir, mode='test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 获取已保存模型文件列表，并按名称排序
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
model_files.sort()

# 3D t-SNE 可视化函数
def visualize_embeddings_3d(embeddings, labels, model_name):
    # 使用 t-SNE 将嵌入降维至 3D 以进行可视化
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # 创建 3D 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 按照标签分别绘制散点图
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], reduced_embeddings[indices, 2],
                   label=f'User {label}', s=15)

    # 设置绘图标题和坐标轴标签
    ax.set_title(f't-SNE 3D Visualization of Embeddings for {model_name}')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_zlabel('t-SNE Dimension 3')
    plt.legend()
    plt.show()

# 打开结果文件，准备写入模型评估结果
with open(result_file, 'w') as f:
    # 遍历所有保存的模型文件
    for model_name in model_files:
        # 加载模型并设置为评估模式
        model = HeartbeatResNet().cuda()  # 将模型移到 GPU 上
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))  # 加载模型权重
        model.eval()  # 设置模型为评估模式

        # 用于存储嵌入和标签的列表
        embeddings = []
        labels = []
        
        # 关闭梯度计算，进行推理
        with torch.no_grad():
            for data in test_loader:
                inputs, labels_batch = data  # 获取输入数据和标签
                inputs = inputs.cuda()  # 将输入数据移到 GPU 上
                outputs = model(inputs)  # 前向传播计算输出
                embeddings.append(outputs.cpu().numpy())  # 将输出转换为 numpy 数组并存储
                labels.extend(labels_batch)  # 存储标签

        embeddings = np.concatenate(embeddings)  # 将所有嵌入拼接成一个数组
        labels = list(map(str, labels))  # 将标签转换为字符串列表

        # 简单的最近邻分类器，计算每个样本的最近邻并预测标签
        preds = []
        for i in range(len(embeddings)):
            # 计算样本 i 与所有嵌入的距离
            distances = F.pairwise_distance(torch.tensor(embeddings[i]).unsqueeze(0), torch.tensor(embeddings))
            distances[i] = float('inf')  # 排除自身距离
            nn_idx = distances.argmin().item()  # 获取最近邻的索引
            preds.append(labels[nn_idx])  # 使用最近邻的标签作为预测结果

        # 计算各项评估指标：准确率、加权 F1 分数、召回率、精确率
        acc = accuracy_score(labels, preds)  # 计算准确率
        f1 = f1_score(labels, preds, average='weighted')  # 计算加权 F1 分数
        recall = recall_score(labels, preds, average='weighted')  # 计算加权召回率
        precision = precision_score(labels, preds, average='weighted')  # 计算加权精确率
        
        # 将模型名称、准确率、F1 分数、召回率、精确率写入结果文件
        f.write(f'{model_name}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}\n')
        print(f'Tested {model_name}: Accuracy={acc:.4f}, F1 Score={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}')
        
        # 对嵌入进行 3D 可视化
        visualize_embeddings_3d(embeddings, labels, model_name)
