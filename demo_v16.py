"""auto-repair-mujoco V16 demo: Obstacle Avoidance 避障到达任务"""
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

# 避障路径：起点 → 绕到障碍物上方 → 绕下来 → 终点
STAGES = [
    ([0.5, 0.5], [-0.5, 0.5]),     # 1. 移动到左上方
    ([-0.5, 0.5], [-0.5, -0.3]),   # 2. 下降到左下方（绕开障碍物左边）
    ([-0.5, -0.3], [0.5, -0.3]),   # 3. 水平移动到右下方（障碍物下方）
    ([0.5, -0.3], [0.5, 0.5]),     # 4. 上升到右上方（绕开障碍物右边）
    ([0.5, 0.5], [0.0, 0.3]),      # 5. 移动到目标点
]

OBSTACLE_POS = np.array([0.0, 0.0])
OBSTACLE_SIZE = 0.15
TARGET_POS = np.array([0.0, 0.3])

def gen_traj(stages, steps=15):
    traj = []
    for s, e in stages:
        s, e = np.array(s), np.array(e)
        for i in range(steps):
            a = i / steps
            traj.append((1 - a) * s + a * e)
    traj.append(np.array(stages[-1][1]))
    return np.array(traj)

print("=" * 60)
print("[V16] Obstacle Avoidance 避障到达任务")
print("=" * 60)

demo = gen_traj(STAGES, 15)
T = len(demo)
act_dim = 2
print(f"阶段数：{len(STAGES)}, 轨迹长度：{T}")
print(f"障碍物位置：{OBSTACLE_POS}, 大小：{OBSTACLE_SIZE}")
print(f"目标位置：{TARGET_POS}")
print(f"问题类型：运动规划（避障）")

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

print("\n训练 Transformer...")
device = 'cpu'

class Policy(nn.Module):
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

model = Policy().to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()
target = torch.tensor(all_traj_n, dtype=torch.float32)
start_t = torch.tensor(start, dtype=torch.float32)

for ep in range(800):
    pred = model(start_t)
    loss = loss_fn(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 200 == 0:
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

# 画障碍物
rect = plt.Rectangle(
    (OBSTACLE_POS[0] - OBSTACLE_SIZE/2, OBSTACLE_POS[1] - OBSTACLE_SIZE/2),
    OBSTACLE_SIZE, OBSTACLE_SIZE,
    color='gray', alpha=0.5, label='Obstacle'
)
ax.add_patch(rect)

# 画目标
ax.scatter(TARGET_POS[0], TARGET_POS[1], c='green', s=200, marker='*', label='Target')

ax.set_title(f'Obstacle Avoidance Task\nFinal err: {np.mean(final_errs)*1000:.1f}mm')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.7, 0.7)
ax.set_ylim(-0.5, 0.7)

ax = axes[1]
errs = np.linalg.norm(pp - tp, axis=1) * 1000
ax.plot(errs, 'g-', linewidth=2)
ax.axhline(y=np.mean(errs), color='r', linestyle='--')
ax.set_title('Per-Step Error')
ax.set_xlabel('Step')
ax.set_ylabel('Error (mm)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v16_obstacle_avoidance.png', dpi=150, bbox_inches='tight')
print("已保存：v16_obstacle_avoidance.png")
print("\nV16 完成！")
print(f"问题类型覆盖：操作任务 + 运动规划（避障）")
