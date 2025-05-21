import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm, trange

# ———— 配置区 ————
GRAPHS_DIR  = Path("D:/电导率预测/processed_graphs")
TARGET_CSV  = Path("extracted_conductivity.csv")
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 100
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
SEED        = 42
EPSILON     = 1e-8     # 防止 log(0)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ——————————

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 1. 读取并变换标签：取自然对数
df = pd.read_csv(TARGET_CSV, dtype={"ID": str})
df = df.dropna(subset=["Ionic conductivity (S cm-1)"])
df["y_raw"] = df["Ionic conductivity (S cm-1)"].astype(float)
df["y"] = np.log(df["y_raw"] + EPSILON)
target_map = dict(zip(df["ID"], df["y"]))

# 2. Dataset 定义
class ConductivityDataset(InMemoryDataset):
    def __init__(self, root, ids):
        super().__init__(root)
        self.ids = ids
        self.data_list = [self._load_and_attach(i) for i in ids]
        self.data, self.slices = self.collate(self.data_list)

    def _load_and_attach(self, idx: str):
        data = torch.load(str(Path(self.root)/f"{idx}.pt"))
        # 将 y 存为一维张量
        data.y = torch.tensor([target_map[idx]], dtype=torch.float)
        return data

    def len(self):
        return len(self.ids)

    def get(self, idx):
        return self.data_list[idx]

# 3. 划分 ID
all_ids = sorted(p.stem for p in GRAPHS_DIR.glob("*.pt") if p.stem in target_map)
random.shuffle(all_ids)
n = len(all_ids)
n_train = int(n * TRAIN_RATIO)
n_val   = int(n * VAL_RATIO)
train_ids = all_ids[:n_train]
val_ids   = all_ids[n_train:n_train+n_val]
test_ids  = all_ids[n_train+n_val:]

# 构建 Dataset
train_ds = ConductivityDataset(GRAPHS_DIR, train_ids)
val_ds   = ConductivityDataset(GRAPHS_DIR, val_ids)
test_ds  = ConductivityDataset(GRAPHS_DIR, test_ids)

# 4. 计算标准化参数（仅用训练集）
# 汇总所有节点特征 x 和边特征 edge_attr
all_x = torch.cat([data.x for data in train_ds.data_list], dim=0)
all_e = torch.cat([data.edge_attr for data in train_ds.data_list], dim=0)
x_mean, x_std = all_x.mean(dim=0), all_x.std(dim=0) + 1e-6
e_mean, e_std = all_e.mean(dim=0), all_e.std(dim=0) + 1e-6

print(f"Node feature mean/std: {x_mean.tolist()}, {x_std.tolist()}")
print(f"Edge feature mean/std: {e_mean.tolist()}, {e_std.tolist()}")

# 5. 应用标准化到所有数据集
def normalize_graph(data):
    data.x = (data.x - x_mean) / x_std
    data.edge_attr = (data.edge_attr - e_mean) / e_std
    return data

for ds in (train_ds, val_ds, test_ds):
    for i, g in enumerate(ds.data_list):
        ds.data_list[i] = normalize_graph(g)

# DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# 6. 定义 GNN 回归模型
class GCNRegressor(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(in_ch, hidden_ch)])
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_ch, hidden_ch))
        self.act = nn.ReLU()
        self.lin1 = nn.Linear(hidden_ch, hidden_ch // 2)
        self.lin2 = nn.Linear(hidden_ch // 2, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.act(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.act(self.lin1(x))
        out = self.lin2(x)   # [batch,1]
        return out.view(-1)  # [batch]

# 实例化模型
sample = torch.load(str(GRAPHS_DIR / f"{train_ids[0]}.pt"))
model = GCNRegressor(sample.x.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 7. 训练与评估函数
def train_epoch():
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def evaluate(loader, split="Val"):
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc=split, leave=False):
        batch = batch.to(DEVICE)
        pred = model(batch.x, batch.edge_index, batch.batch)
        ys.append(batch.y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# 8. 训练循环
best_val_mae = float('inf')
for epoch in trange(1, EPOCHS + 1, desc="Epochs"):
    tr_loss = train_epoch()
    val_mae, val_rmse, val_r2 = evaluate(val_loader, "Val")
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), "best_model.pt")
    tqdm.write(f"Epoch {epoch:3d} | Train Loss: {tr_loss:.4f} | "
               f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")

# 9. 测试评估
model.load_state_dict(torch.load("best_model.pt"))
test_mae, test_rmse, test_r2 = evaluate(test_loader, "Test")
print(f"\nTest MAE (log scale): {test_mae:.4f} | Test RMSE (log scale): {test_rmse:.4f} | Test R²: {test_r2:.4f}")

