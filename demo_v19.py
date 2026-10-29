"""auto-repair-mujoco V19 demo: Vision-guided Reaching 视觉引导到达任务"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

L1, L2 = 0.3, 0.25

def fk(theta):
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# 视觉引导：目标位置随机，机器人要从初始位置到达目标
# 区别：之前的任务轨迹是固定的，这个任务的目标是随机的，要泛化
def gen_reach_traj(start_xy, target_xy, steps=30):
    """生成从start到target的直线轨迹"""
    traj = []
    for i in range(steps):
        a = i / steps
        traj.append((1 - a) * np.array(start_xy) + a * np.array(target_xy))
    return np.array(traj)

print("=" * 60)
print("[V19] Vision-guided Reaching 视觉引导到达任务")
print("=" * 60)

# 生成多条不同目标的轨迹，测试泛化能力
np.random.seed(42)
n_tasks = 20  # 20个不同的目标位置
all_trajs = []
all_targets = []

for _ in range(n_tasks):
    target = np.random.uniform(-0.4, 0.4, size=2)
    start = np.array([0.0, 0.5])
    traj = gen_reach_traj(start, target, steps=30)
    traj += np.random.normal(0, 0.002, traj.shape)
    all_trajs.append(traj)
    all_targets.append(target)

all_trajs = np.array(all_trajs)  # (n_tasks, 30, 2)
all_targets = np.array(all_targets)  # (n_tasks, 2)

T = 30
act_dim = 2
obs_dim = 2  # 观测：目标位置（视觉输入）

print(f"任务数量：{n_tasks} 个不同目标")
print(f"轨迹长度：{T} 步")
print(f"观测维度：{obs_dim}（目标位置坐标，模拟视觉输入）")
print(f"动作维度：{act_dim}（末端位置）")
print(f"测试能力：泛化到新目标位置")

# 标准化
mean = all_trajs.mean(axis=(0, 1))
std = all_trajs.std(axis=(0, 1)) + 1e-6
all_trajs_n = (all_trajs - mean) / std

target_mean = all_targets.mean(axis=0)
target_std = all_targets.std(axis=0) + 1e-6
all_targets_n = (all_targets - target_mean) / target_std

# 训练集：15个任务
train_trajs = all_trajs_n[:15]
train_targets = all_targets_n[:15]
# 测试集：5个新任务（没见过的目标）
test_trajs = all_trajs_n[15:]
test_targets = all_targets_n[15:]

print(f"\n训练集：{len(train_trajs)} 个任务")
print(f"测试集：{len(test_trajs)} 个新任务（测试泛化）")

print("\n训练 Transformer...")
device = 'cpu'

class VisionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # 视觉编码器：把目标位置编码成特征
        self.visual_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 96)
        )
        # 位置编码
        self.pos = nn.Parameter(torch.randn(1, T, 96) * 0.02)
        # Transformer decoder
        layer = nn.TransformerDecoderLayer(d_model=96, nhead=4, dim_feedforward=192, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=3)
        # 动作头
        self.head = nn.Linear(96, act_dim)
        
    def forward(self, target_obs):
        B = target_obs.shape[0]
        # 视觉编码作为memory
        memory = self.visual_encoder(target_obs)  # (B, 96)
        memory = memory.unsqueeze(1)  # (B, 1, 96)
        # query是位置编码
        query = self.pos.expand(B, -1, -1)  # (B, T, 96)
        out = self.dec(query, memory)
        return self.head(out)

model = VisionPolicy().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

train_target_t = torch.tensor(train_trajs, dtype=torch.float32)
train_obs_t = torch.tensor(train_targets, dtype=torch.float32)

for ep in range(1000):
    pred = model(train_obs_t)
    loss = loss_fn(pred, train_target_t)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 250 == 0:
        print(f"  epoch {ep+1}, loss={loss.item():.6f}")

print("\n测试泛化能力（测试集）...")
model.eval()
with torch.no_grad():
    test_pred = model(torch.tensor(test_targets, dtype=torch.float32))

test_pred_np = test_pred.numpy() * std + mean
test_true_np = test_trajs * std + mean

final_errs, traj_errs = [], []
for i in range(len(test_pred_np)):
    pp = np.array([fk(t) for t in test_pred_np[i]])
    tp = np.array([fk(t) for t in test_true_np[i]])
    final_errs.append(np.linalg.norm(pp[-1] - tp[-1]))
    traj_errs.append(np.mean(np.linalg.norm(pp - tp, axis=1)))

print(f"测试集最终误差：{np.mean(final_errs)*1000:.1f} mm")
print(f"测试集平均轨迹误差：{np.mean(traj_errs)*1000:.1f} mm")

print("\n生成图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：测试集的3个样本
ax = axes[0]
colors = ['b', 'g', 'r']
for i in range(3):
    tp = np.array([fk(t) for t in test_true_np[i]])
    pp = np.array([fk(t) for t in test_pred_np[i]])
    ax.plot(tp[:, 0], tp[:, 1], colors[i]+'-', label=f'True {i+1}', linewidth=2)
    ax.plot(pp[:, 0], pp[:, 1], colors[i]+'--', label=f'Pred {i+1}', linewidth=2)
    # 画目标点
    target_pos = fk(np.array([test_trajs[i][-1]]))[0]
    ax.scatter(target_pos[0], target_pos[1], c=colors[i], s=100, marker='*')

ax.set_title(f'Vision-guided Reaching (Generalization)\nAvg final err: {np.mean(final_errs)*1000:.1f}mm')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(-0.2, 0.7)

# 右图：每步误差（所有测试样本平均）
ax = axes[1]
all_errs = []
for i in range(len(test_pred_np)):
    pp = np.array([fk(t) for t in test_pred_np[i]])
    tp = np.array([fk(t) for t in test_true_np[i]])
    errs = np.linalg.norm(pp - tp, axis=1) * 1000
    all_errs.append(errs)
all_errs = np.array(all_errs)
mean_errs = all_errs.mean(axis=0)
std_errs = all_errs.std(axis=0)

steps = np.arange(T)
ax.plot(steps, mean_errs, 'g-', linewidth=2, label='Mean error')
ax.fill_between(steps, mean_errs - std_errs, mean_errs + std_errs, alpha=0.3, color='g')
ax.set_title('Per-Step Error (Test Set)')
ax.set_xlabel('Step')
ax.set_ylabel('Error (mm)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v19_vision_guided.png', dpi=150, bbox_inches='tight')
print("已保存：v19_vision_guided.png")
print("\nV19 完成！")
print(f"能力覆盖：固定轨迹 + 泛化到新目标（视觉引导）")
