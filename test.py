# test.py
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataloader import HeartbeatDataset
from models import HeartbeatResNet

test_dir = 'test_data/device3/p4-test/'
model_dir = 'save_models'
result_file = 'result.txt'

# 获取所有用户列表并选择最后4个用户进行测试
all_users = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
all_users.sort()
test_users = all_users[8:]

# 加载测试数据集
test_dataset = HeartbeatDataset(test_dir, mode='test', user_list=test_users)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 列出已保存的模型
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
model_files.sort()

with open(result_file, 'w') as f:
    for model_name in model_files:
        # 加载模型
        model = HeartbeatResNet().cuda()
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
        model.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for data in test_loader:
                inputs, labels_batch = data  # 解包输入和标签
                inputs = inputs.cuda()
                outputs = model(inputs)
                embeddings.append(outputs.cpu())
                labels.extend(labels_batch)
        embeddings = torch.cat(embeddings)
        # 将标签转换为列表（如果还没有）
        labels = list(map(str, labels))
        # 简单的最近邻分类
        preds = []
        for i in range(len(embeddings)):
            distances = F.pairwise_distance(embeddings[i].unsqueeze(0), embeddings)
            distances[i] = float('inf')  # 排除自身
            nn_idx = distances.argmin().item()
            preds.append(labels[nn_idx])
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='weighted')
        # 写入结果
        f.write(f'{model_name}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}\n')
        print(f'Tested {model_name}: Accuracy={acc:.4f}, F1 Score={f1:.4f}')
