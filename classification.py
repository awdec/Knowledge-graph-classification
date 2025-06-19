import os
import csv
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载文本编码器
encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 标签映射
label_map = {"AAAI": 0, "NeurIPS": 1, "CVPR": 2}
# label_map = {"AAAI": 0, "NeurIPS": 1}
subdirs = list(label_map.keys())

print(subdirs)

root_dir = "./data_output/cureus"


def encode_text_list(texts):
    embeddings = encoder.encode(texts, convert_to_tensor=True)
    return embeddings


num = [0, 0, 0]


def csv_to_pyg_data(csv_path, graph_label):
    """
    1. 用 csv.reader 读取前 3 列：node_1, node_2, edge
    2. 跳过列数不足 3 的行，忽略第 4 列或更多列
    3. 构建图数据对象返回
    """
    print(f"正在处理文件：{csv_path}")

    if num[graph_label] == 200:
        print("样本" + str(graph_label) + "已够 200")
        return None

    valid_rows = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            if len(header) < 3:
                print(f"[警告] 文件 header 列数为 {len(header)}，少于 3")
                return None
        except StopIteration:
            print(f"[警告] 文件为空或无法读取 header：{csv_path}")
            return None

        for lineno, row in enumerate(reader, start=2):
            if len(row) < 3:
                continue
            # 只取前 3 列
            valid_rows.append(row[:3])

    if len(valid_rows) == 0:
        print(f"[警告] 过滤后没有合法行，文件：{csv_path}")
        return None

    num[graph_label] += 1

    df = pd.DataFrame(valid_rows, columns=['node_1', 'node_2', 'edge'])

    nodes = pd.unique(df[['node_1', 'node_2']].values.ravel())
    node2id = {name: i for i, name in enumerate(nodes)}

    x = encode_text_list(list(nodes))

    edge_index = []
    edge_attr_text = []
    for _, row in df.iterrows():
        u = node2id[row['node_1']]
        v = node2id[row['node_2']]
        edge_index.append([u, v])
        edge_attr_text.append(row['edge'])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = encode_text_list(edge_attr_text)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([graph_label], dtype=torch.long),
    )


# ========================= 主流程 ========================= #
data_list = []
for subdir in subdirs:
    path = os.path.join(root_dir, subdir)
    if not os.path.isdir(path):
        continue
    for fname in os.listdir(path):
        if not fname.endswith(".csv"):
            continue
        csv_path = os.path.join(path, fname)
        data = csv_to_pyg_data(csv_path, label_map[subdir])
        if data is None:
            continue
        data_list.append(data)

if len(data_list) == 0:
    raise RuntimeError("没有读取到任何合法的图数据，请检查 CSV 文件。")

# ✅ 新增：打乱数据顺序
import random

random.seed(int(time.time()))
random.shuffle(data_list)

# ✅ 原有：按比例划分数据集
train_data, temp_data = train_test_split(
    data_list,
    test_size=0.3,
    stratify=[d.y.item() for d in data_list],
    random_state=int(time.time()),
)
val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    stratify=[d.y.item() for d in temp_data],
    random_state=int(time.time()),
)

from collections import Counter


def print_label_distribution(name, dataset):
    labels = [d.y.item() for d in dataset]
    counter = Counter(labels)
    print(f"\n{name} 样本数: {len(labels)}")
    for label, count in counter.items():
        print(f"  标签 {label}：{count} 个样本")


# 打印每个子集的信息
print_label_distribution("训练集", train_data)
print_label_distribution("验证集", val_data)
print_label_distribution("测试集", test_data)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=16)


class GINE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINE, self).__init__()

        self.edge_lin = torch.nn.Linear(in_channels, hidden_channels)

        self.node_mlp1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.node_mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )

        self.conv1 = GINEConv(self.node_mlp1, edge_dim=hidden_channels)
        self.conv2 = GINEConv(self.node_mlp2, edge_dim=hidden_channels)

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        # x:          [num_nodes,    in_channels]   （比如 [N,384]）
        # edge_attr:  [num_edges,    in_channels]   （比如 [E,384]）
        # batch:      [num_nodes]   （batch 指示每个节点属于哪个图）

        # —— 1. 先把边特征从 in_channels → hidden_channels —— #
        edge_attr_proj = self.edge_lin(edge_attr)  # 结果 [num_edges, hidden_channels]

        # —— 2. 第一层消息传递 —— #
        x = self.conv1(x, edge_index, edge_attr_proj)  # 输出: [num_nodes, hidden_channels]
        x = F.relu(x)

        # —— 3. 第二层消息传递 —— #
        x = self.conv2(x, edge_index, edge_attr_proj)  # 输出: [num_nodes, hidden_channels]

        # —— 4. 全局池化，把每张图的所有节点聚成一个向量 —— #
        x = global_mean_pool(x, batch)  # 结果: [num_graphs, hidden_channels]

        # —— 5. 最后一个线性层映射到 out_channels —— #
        x = self.lin(x)  # 结果: [num_graphs, out_channels]
        return x


model = GINE(
    in_channels=encoder.get_sentence_embedding_dimension(),  # 例如 384
    hidden_channels=64,
    out_channels=len(label_map),  # 例如 3
).to(device)

learn = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = F.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        all_preds.append(pred.cpu())
        all_labels.append(batch.y.view(-1).cpu())
        all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_prob = torch.cat(all_probs).numpy()

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    try:
        if y_prob.shape[1] == 2:
            # 二分类：只用类别 1 的概率
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            # 多分类：使用 one-vs-rest 策略
            roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        roc_auc = float("nan")

    return acc, precision, recall, f1, roc_auc, y_true, y_prob


# ===== 训练主循环 ===== #
best_val_acc = 0.0
patience = 10
counter = 0
train_losses = []
val_accuracies = []

for epoch in range(1, 51):
    loss = train()
    acc, precision, recall, f1, *_ = evaluate(val_loader)
    train_losses.append(loss)
    val_accuracies.append(acc)
    print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    if acc > best_val_acc:
        best_val_acc = acc
        counter = 0
        torch.save(model.state_dict(), "best_gine_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("验证集性能未提升，早停。")
            break
            # counter = 0
            # if learn < 0.0005:
            #     print("验证集性能未提升，早停。")
            #     break
            # learn = learn / 2.0
            optimizer = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=1e-4)

# ...（前面代码保持不变，直到 evaluate 和 ROC 绘图部分）

# ===== 测试集评估 ===== #
model.load_state_dict(torch.load("best_gine_model.pth"))
acc, prec, rec, f1, roc_auc, y_true, y_prob = evaluate(test_loader)

print("\n=== 最终测试集评估结果 ===")
print(f"准确率（Accuracy）  : {acc:.4f}")
print(f"精确率（Precision）  : {prec:.4f}")
print(f"召回率（Recall）    : {rec:.4f}")
print(f"F1 分数           : {f1:.4f}")
print(f"ROC AUC（宏平均） : {roc_auc:.4f}")

# ===== 绘制 Loss 曲线 ===== #
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# ===== 绘制 ROC 曲线（兼容二分类和多分类） ===== #
from sklearn.preprocessing import label_binarize

n_classes = len(label_map)
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

if n_classes == 2:
    # 二分类，只使用正类的概率（列索引为 1）
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"Class 1 (AUC = {roc_auc_val:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()
else:
    # 多分类
    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7, 5))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc_dict[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("roc_curve.png")
    plt.show()
