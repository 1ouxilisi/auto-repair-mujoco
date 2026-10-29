"""auto-repair-mujoco V18 demo: Multi-step Assembly 多步装配序列"""
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

# 多步装配：装3个零件，按顺序
STAGES = [
    # 第一步：装零件1
    ([0.5, 0.5], [-0.5, 0.3]),     # 1. 移动到零件1上方
    ([-0.5, 0.3], [-0.5, 0.0]),    # 2. 下降抓零件1
    ([-0.5, 0.0], [-0.5, 0.2]),    # 3. 拿起零件1
    ([-0.5, 0.2], [-0.2, 0.2]),   # 4. 移动到装配位1
    ([-0.2, 0.2], [-0.2, 0.0]),   # 5. 下降装配零件1
    # 第二步：装零件2
    ([-0.2, 0.0], [0.0, 0.3]),    # 6. 移动到零件2上方
    ([0.0, 0.3], [0.0, 0.0]),      # 7. 下降抓零件2
    ([0.0, 0.0], [0.0, 0.2]),      # 8. 拿起零件2
    ([0.0, 0.2], [0.0, 0.2]),      # 9. 移动到装配位2（就在旁边）
    ([0.0, 0.2], [0.0, 0.0]),      # 10. 下降装配零件2
    # 第三步：装零件3
    ([0.0, 0.0], [0.5, 0.3]),      # 11. 移动到零件3上方
    ([0.5, 0.3], [0.5, 0.0]),      # 12. 下降抓零件3
    ([0.5, 0.0], [0.5, 0.2]),      # 13. 拿起零件3
    ([0.5, 0.2], [0.2, 0.2]),      # 14. 移动到装配位3
    ([0.2, 0.2], [0.2, 0.0]),      # 15. 下降装配零件3
    # 收回
    ([0.2, 0.0], [0.5, 0.5]),      # 16. 收回
]

PART1_POS = np.array([-0.5, 0.0])
PART2_POS = np.array([0.0, 0.0])
PART3_POS = np.array([0.5, 0.0])
ASSEMBLY_POS = np.array([0.0, 0.0])

def gen_traj(stages, steps=8):
    traj = []
    for s, e in stages:
        s, e = np.array(s), np.array(e)
        for i in range(steps):
            a = i / steps
            traj.append((1 - a) * s + a * e)
    traj.append(np.array(stages[-1][1]))
    return np.array(traj)

print("=" * 60)
print("[V18] Multi-step Assembly 多步装配序列")
print("=" * 60)

demo = gen_traj(STAGES, 8)
T = len(demo)
act_dim = 2
print(f"阶段数：{len(STAGES)}, 轨迹长度：{T}")
print(f"零件1位置：{PART1_POS}")
print(f"零件2位置：{PART2_POS}")
print(f"零件3位置：{PART3_POS}")
print(f"装配位置：{ASSEMBLY_POS}")
print(f"任务类型：多步装配序列（3个零件按顺序）")

np.random.seed(42)
n = 10
all_traj = []
for _ in range(n):
    all_traj.append(demo + np.random.normal(0, 0.002, demo.shape))
all_traj = np.array(all_traj)

mean = all_traj.mean(axis=(0, 1))
std = all_traj.std(axis=(0, 1)) + 1e-6
all_traj_n = (all_traj - mean) / std
start = all_traj_n[:, 0:1, :]

print("\n训练 Transformer...")
device = 'cpu'

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(act_dim, 96)
        self.pos = nn.Parameter(torch.randn(1, T, 96) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=96, nhead=4, dim_feedforward=192, batch_first=True)
        self.dec = nn.TransformerDecoder(layer, num_layers=4)
        self.head = nn.Linear(96, act_dim)
    def forward(self, x):
        B = x.shape[0]
        mem = self.inp(x)
        q = self.pos.expand(B, -1, -1)
        return self.head(self.dec(q, mem))

model = Policy().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()
target = torch.tensor(all_traj_n, dtype=torch.float32)
start_t = torch.tensor(start, dtype=torch.float32)

for ep in range(1000):
    pred = model(start_t)
    loss = loss_fn(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 250 == 0:
        print(f"  epoch {ep+1}, loss={loss.item():.6f}")

print("\n验证...")
model.eval()
with torch.no_grad():
    pred = model(start_t)
pred_np = pred.numpy() * std + mean

final_errs, traj_errs = [], []
for i in range(n):
    pp = np.array([fk(t) for t in pred_np[i]])
    tp = np.array([fk(t) for t in all_traj[i]])
    final_errs.append(np.linalg.norm(pp[-1] - tp[-1]))
    traj_errs.append(np.mean(np.linalg.norm(pp - tp, axis=1)))

print(f"最终误差：{np.mean(final_errs)*1000:.1f} mm")
print(f"平均轨迹误差：{np.mean(traj_errs)*1000:.1f} mm")

print("\n生成图...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ti = 0
tp = np.array([fk(t) for t in all_traj[ti]])
pp = np.array([fk(t) for t in pred_np[ti]])

ax = axes[0]
ax.plot(tp[:, 0], tp[:, 1], 'b-', label='True', linewidth=2)
ax.plot(pp[:, 0], pp[:, 1], 'r--', label='Pred', linewidth=2)

# 画三个零件
ax.scatter(PART1_POS[0], PART1_POS[1], c='green', s=150, marker='o', label='Part 1')
ax.scatter(PART2_POS[0], PART2_POS[1], c='orange', s=150, marker='o', label='Part 2')
ax.scatter(PART3_POS[0], PART3_POS[1], c='purple', s=150, marker='o', label='Part 3')

# 画装配位置
ax.scatter(ASSEMBLY_POS[0], ASSEMBLY_POS[1], c='red', s=200, marker='*', label='Assembly Target')

ax.set_title(f'Multi-step Assembly (3 Parts)\nFinal err: {np.mean(final_errs)*1000:.1f}mm')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.3, 0.7)

ax = axes[1]
errs = np.linalg.norm(pp - tp, axis=1) * 1000
ax.plot(errs, 'g-', linewidth=2)
ax.axhline(y=np.mean(errs), color='r', linestyle='--')
ax.set_title('Per-Step Error')
ax.set_xlabel('Step')
ax.set_ylabel('Error (mm)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v18_multi_step_assembly.png', dpi=150, bbox_inches='tight')
print("已保存：v18_multi_step_assembly.png")
print("\nV18 完成！")
print(f"装配任务覆盖：单步装配(Assembly) + 多步序列装配(Multi-step Assembly)")
