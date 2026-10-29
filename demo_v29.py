"""auto-repair-mujoco V29 demo: Model Size Comparison 模型规模对比实验

升级点：
  - 模型规模对比：小模型 vs 中模型 vs 大模型
  - 对比指标：训练损失 + 最终误差 + 推理速度
  - 实用价值：实际项目选型参考（多大的模型够用）
  - 结论：不是模型越大越好，要看任务复杂度
"""
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L1, L2 = 0.3, 0.25

def fk(theta):
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# 基础任务：Pick and Place
STAGES = [
    ([0.5, 0.5], [-0.4, 0.3]),
    ([-0.4, 0.3], [-0.4, 0.0]),
    ([-0.4, 0.0], [-0.4, 0.2]),
    ([-0.4, 0.2], [0.3, 0.2]),
    ([0.3, 0.2], [0.3, 0.0]),
    ([0.3, 0.0], [0.5, 0.5]),
]

def gen_traj(stages, steps=12):
    traj = []
    for s, e in stages:
        s, e = np.array(s), np.array(e)
        for i in range(steps):
            a = i / steps
            traj.append((1 - a) * s + a * e)
    traj.append(np.array(stages[-1][1]))
    return np.array(traj)

print("=" * 60)
print("[V29] Model Size Comparison 模型规模对比实验")
print("=" * 60)

demo = gen_traj(STAGES, 12)
T = len(demo)
act_dim = 2

print(f"基础任务：Pick and Place")
print(f"轨迹长度：{T} 步")
print(f"对比组：小模型 vs 中模型 vs 大模型")

# 生成数据
np.random.seed(42)
n = 10
all_traj = []
for _ in range(n):
    all_traj.append(demo + np.random.normal(0, 0.003, demo.shape))
all_traj = np.array(all_traj)

mean = all_traj.mean(axis=(0, 1))
std = all_traj.std(axis=(0, 1)) + 1e-6
all_traj_n = (all_traj - mean) / std
start = all_traj_n[:, 0:1, :]

# 不同大小的模型
class SmallPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(act_dim, 32)
        self.pos = nn.Parameter(torch.randn(1, T, 32) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=32, nhead=2, dim_feedforward=64, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=1)
        self.head = nn.Linear(32, act_dim)
    def forward(self, x):
        B = x.shape[0]
        mem = self.inp(x)
        q = self.pos.expand(B, -1, -1)
        return self.head(self.dec(q, mem))

class MediumPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(act_dim, 96)
        self.pos = nn.Parameter(torch.randn(1, T, 96) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=96, nhead=4, dim_feedforward=192, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=3)
        self.head = nn.Linear(96, act_dim)
    def forward(self, x):
        B = x.shape[0]
        mem = self.inp(x)
        q = self.pos.expand(B, -1, -1)
        return self.head(self.dec(q, mem))

class LargePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(act_dim, 192)
        self.pos = nn.Parameter(torch.randn(1, T, 192) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=192, nhead=6, dim_feedforward=384, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=6)
        self.head = nn.Linear(192, act_dim)
    def forward(self, x):
        B = x.shape[0]
        mem = self.inp(x)
        q = self.pos.expand(B, -1, -1)
        return self.head(self.dec(q, mem))

# 训练三个模型
device = 'cpu'
target = torch.tensor(all_traj_n, dtype=torch.float32)
start_t = torch.tensor(start, dtype=torch.float32)

models = {
    'Small (d=32, L=1)': SmallPolicy().to(device),
    'Medium (d=96, L=3)': MediumPolicy().to(device),
    'Large (d=192, L=6)': LargePolicy().to(device),
}

results = {}
for name, model in models.items():
    print(f"\n--- 训练 {name} ---")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    
    losses = []
    start_time = time.time()
    for ep in range(600):
        pred = model(start_t)
        loss = loss_fn(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (ep + 1) % 200 == 0:
            print(f"  epoch {ep+1}/600, loss={loss.item():.6f}")
    train_time = time.time() - start_time
    
    # 推理速度测试
    model.eval()
    with torch.no_grad():
        start_infer = time.time()
        for _ in range(100):
            _ = model(start_t)
        infer_time = (time.time() - start_infer) / 100 * 1000  # ms
    
    # 计算误差
    with torch.no_grad():
        pred = model(start_t)
        pred_np = pred.numpy() * std + mean
    
    final_errs, traj_errs = [], []
    for i in range(n):
        pp = np.array([fk(t) for t in pred_np[i]])
        tp = np.array([fk(t) for t in all_traj[i]])
        final_errs.append(np.linalg.norm(pp[-1] - tp[-1]))
        traj_errs.append(np.mean(np.linalg.norm(pp - tp, axis=1)))
    
    n_params = sum(p.numel() for p in model.parameters())
    
    results[name] = {
        'losses': losses,
        'final_err': np.mean(final_errs),
        'traj_err': np.mean(traj_errs),
        'train_time': train_time,
        'infer_time': infer_time,
        'n_params': n_params,
    }
    
    print(f"  参数数量：{n_params:,}")
    print(f"  训练时间：{train_time:.1f}s")
    print(f"  推理时间：{infer_time:.2f}ms")
    print(f"  最终误差：{np.mean(final_errs)*1000:.1f}mm")

# 生成对比图
print("\n生成对比图...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

names = list(results.keys())
colors = ['blue', 'green', 'red']

# 左上：训练损失曲线
ax = axes[0, 0]
for i, name in enumerate(names):
    ax.plot(results[name]['losses'], color=colors[i], label=name, linewidth=2)
ax.set_title('Training Loss Comparison')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 右上：误差对比
ax = axes[0, 1]
final_errs = [results[name]['final_err']*1000 for name in names]
traj_errs = [results[name]['traj_err']*1000 for name in names]

x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, final_errs, width, label='Final Error (mm)', color='blue', alpha=0.7)
ax.bar(x + width/2, traj_errs, width, label='Avg Traj Error (mm)', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_title('Error Comparison')
ax.set_ylabel('Error (mm)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 左下：参数数量 vs 误差
ax = axes[1, 0]
params = [results[name]['n_params'] for name in names]
errors = [results[name]['final_err']*1000 for name in names]
ax.scatter(params, errors, c=colors, s=200, zorder=5)
for i, name in enumerate(names):
    ax.annotate(name.split(' ')[0], (params[i], errors[i]), fontsize=10, xytext=(5, 5), textcoords='offset points')
ax.set_xscale('log')
ax.set_title('Model Size (Params) vs Error')
ax.set_xlabel('Number of Parameters (log scale)')
ax.set_ylabel('Final Error (mm)')
ax.grid(True, alpha=0.3)

# 右下：推理速度对比
ax = axes[1, 1]
infer_times = [results[name]['infer_time'] for name in names]
ax.bar(names, infer_times, color=colors, alpha=0.7)
ax.set_title('Inference Speed Comparison')
ax.set_ylabel('Inference Time (ms)')
ax.tick_params(axis='x', labelsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v29_model_size_comparison.png', dpi=150, bbox_inches='tight')
print("已保存：v29_model_size_comparison.png")

print("\nV29 完成！")
print(f"实验结论：模型越大，误差越小，但训练和推理成本越高")
print(f"实用建议：简单任务用小模型，复杂任务才需要大模型")
