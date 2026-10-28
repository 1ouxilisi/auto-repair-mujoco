"""
auto-repair-mujoco V4 demo：12阶段完整电池包维修流程

升级点：
  - 从 8 阶段升级到 12 阶段（更完整的维修流程）
  - 更大的 Transformer 模型（d_model=128, 4层 decoder）
  - 更好的可视化（4合1 详细图）
  - 更完整的评估指标
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 1. 机械臂 + 任务定义 ============
L1, L2 = 0.3, 0.25

def forward_kinematics(theta):
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# 12阶段完整电池包维修流程
STAGES = [
    ("1. 移动到电池包上方", [0.5, 0.5], [-0.5, 0.5]),
    ("2. 下降到电池包表面", [-0.5, 0.5], [-0.5, 1.2]),
    ("3. 转向验电笔方向", [-0.5, 1.2], [-0.1, 1.0]),
    ("4. 移动到验电笔上方", [-0.1, 1.0], [0.3, 0.9]),
    ("5. 拿起验电笔",       [0.3, 0.9], [0.3, 1.2]),
    ("6. 移动到检测点A",     [0.3, 1.2], [-0.3, 1.3]),
    ("7. 验电检测A",         [-0.3, 1.3], [-0.3, 1.3]),
    ("8. 移动到检测点B",     [-0.3, 1.3], [0.0, 1.4]),
    ("9. 验电检测B",         [0.0, 1.4], [0.0, 1.4]),
    ("10. 放下验电笔",       [0.0, 1.4], [0.3, 0.9]),
    ("11. 移动到扳手位置",   [0.3, 0.9], [-0.2, 0.8]),
    ("12. 收回安全位置",     [-0.2, 0.8], [0.4, 0.6]),
]

def generate_trajectory(stages, steps_per_stage=10):
    traj = []
    for name, start, end in stages:
        start = np.array(start)
        end = np.array(end)
        for s in range(steps_per_stage):
            alpha = s / steps_per_stage
            traj.append((1 - alpha) * start + alpha * end)
    traj.append(np.array(stages[-1][2]))
    return np.array(traj)

# ============ 2. 生成数据 ============
print("=" * 60)
print("[V4] 12阶段完整电池包维修流程")
print("=" * 60)

demo_traj = generate_trajectory(STAGES, steps_per_stage=10)
T = len(demo_traj)
act_dim = 2

print(f"任务阶段数：{len(STAGES)} 个阶段")
print(f"轨迹长度：{T} 步")
print(f"动作维度：{act_dim}（2关节角度）")

# 生成多条带噪声的演示轨迹
np.random.seed(42)
n_demos = 8
all_trajs = []
for _ in range(n_demos):
    noise = np.random.normal(0, 0.003, demo_traj.shape)
    all_trajs.append(demo_traj + noise)
all_trajs = np.array(all_trajs)  # (n_demos, T, act_dim)

print(f"演示轨迹数：{n_demos} 条")

# 标准化
act_mean = all_trajs.mean(axis=(0, 1))
act_std = all_trajs.std(axis=(0, 1)) + 1e-6
all_trajs_n = (all_trajs - act_mean) / act_std

# 起始观测
start_act = all_trajs_n[:, 0:1, :]  # (n_demos, 1, act_dim)

# ============ 3. Transformer 模型 ============
print("\n" + "=" * 60)
print("[1/3] 训练 Transformer 模型")
print("=" * 60)

device = 'cpu'

class TransformerPolicy(nn.Module):
    def __init__(self, act_dim=2, seq_len=121, d_model=128, nhead=4, num_layers=4):
        super().__init__()
        self.seq_len = seq_len
        self.start_proj = nn.Linear(act_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.act_head = nn.Linear(d_model, act_dim)
        
    def forward(self, start_act):
        B = start_act.shape[0]
        memory = self.start_proj(start_act)
        query = self.pos_embed.expand(B, -1, -1)
        out = self.decoder(query, memory)
        return self.act_head(out)

model = TransformerPolicy(act_dim=act_dim, seq_len=T, d_model=128, num_layers=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

target_t = torch.tensor(all_trajs_n, dtype=torch.float32).to(device)
start_t = torch.tensor(start_act, dtype=torch.float32).to(device)

epochs = 1000
for ep in range(epochs):
    pred = model(start_t)
    loss = loss_fn(pred, target_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (ep + 1) % 200 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

# ============ 4. 验证 ============
print("\n" + "=" * 60)
print("[2/3] 验证模型效果")
print("=" * 60)

model.eval()
with torch.no_grad():
    pred = model(start_t)

# 反标准化
pred_np = pred.cpu().numpy() * act_std + act_mean
true_np = all_trajs

# 计算误差
def compute_errors(pred, true):
    final_errs = []
    traj_errs = []
    for i in range(len(pred)):
        pred_pos = np.array([forward_kinematics(theta) for theta in pred[i]])
        true_pos = np.array([forward_kinematics(theta) for theta in true[i]])
        final_err = np.linalg.norm(pred_pos[-1] - true_pos[-1])
        traj_err = np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))
        final_errs.append(final_err)
        traj_errs.append(traj_err)
    return np.mean(final_errs), np.mean(traj_errs)

final_err, traj_err = compute_errors(pred_np, true_np)

print(f"最终位置误差：{final_err*1000:.1f} mm")
print(f"平均轨迹误差：{traj_err*1000:.1f} mm")

# ============ 5. 可视化 ============
print("\n" + "=" * 60)
print("[3/3] 生成可视化图")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

demo_idx = 0
true_traj = true_np[demo_idx]
pred_traj = pred_np[demo_idx]

true_pos = np.array([forward_kinematics(theta) for theta in true_traj])
pred_pos = np.array([forward_kinematics(theta) for theta in pred_traj])

# 左上：完整轨迹
ax = axes[0, 0]
ax.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True', linewidth=2)
ax.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label='Predicted', linewidth=2)
ax.scatter([0], [0], c='k', s=100, marker='s', label='Base')

# 标记每个阶段的起点
for i, (name, start, end) in enumerate(STAGES):
    start_pos = forward_kinematics(np.array(start))
    ax.scatter(start_pos[0], start_pos[1], c='green', s=50, marker='o')
    ax.annotate(str(i+1), (start_pos[0], start_pos[1]), fontsize=8)

ax.set_title(f'12-Stage Full Repair Trajectory\nFinal error: {final_err*1000:.1f}mm, Avg error: {traj_err*1000:.1f}mm')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 右上：关节角度随时间变化
ax = axes[0, 1]
steps = np.arange(T)
ax.plot(steps, true_traj[:, 0], 'b-', label='Theta1 True', linewidth=2)
ax.plot(steps, pred_traj[:, 0], 'r--', label='Theta1 Pred', linewidth=2)
ax.plot(steps, true_traj[:, 1], 'b-', label='Theta2 True', linewidth=1, alpha=0.5)
ax.plot(steps, pred_traj[:, 1], 'r--', label='Theta2 Pred', linewidth=1, alpha=0.5)
ax.set_title('Joint Angles Over Time')
ax.set_xlabel('Step')
ax.set_ylabel('Angle (rad)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 左下：每步误差
ax = axes[1, 0]
step_errs = np.linalg.norm(pred_pos - true_pos, axis=1) * 1000
ax.plot(steps, step_errs, 'g-', linewidth=2)
ax.set_title(f'Per-Step Error (Avg: {np.mean(step_errs):.1f}mm)')
ax.set_xlabel('Step')
ax.set_ylabel('Error (mm)')
ax.grid(True, alpha=0.3)

# 右下：各阶段误差
ax = axes[1, 1]
stages_err = []
steps_per_stage = 10
for i in range(len(STAGES)):
    start_idx = i * steps_per_stage
    end_idx = min(start_idx + steps_per_stage, T)
    stage_err = np.mean(step_errs[start_idx:end_idx])
    stages_err.append(stage_err)

ax.bar(range(1, len(STAGES)+1), stages_err, color='orange', alpha=0.7)
ax.set_title('Error by Stage')
ax.set_xlabel('Stage Number')
ax.set_ylabel('Error (mm)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v4_12stage_repair.png', dpi=150, bbox_inches='tight')
print("可视化图已保存：v4_12stage_repair.png")

# ============ 6. 总结 ============
print("\n" + "=" * 60)
print("V4 Demo 完成！")
print("=" * 60)
print(f"✅ 12阶段完整维修流程（从8阶段升级）")
print(f"✅ 更大的 Transformer 模型（d_model=128, 4层 decoder）")
print(f"✅ 最终位置误差：{final_err*1000:.1f} mm")
print(f"✅ 平均轨迹误差：{traj_err*1000:.1f} mm")
print(f"✅ 4合1 可视化图：v4_12stage_repair.png")
print()
print("版本演进：")
print(f"  V0: 1阶段, MLP, 误差 1mm")
print(f"  V1: 4阶段, Transformer, 误差 1.2mm")
print(f"  V2: 8阶段, 条件Transformer, 误差 2.7mm")
print(f"  V4: 12阶段, 大Transformer, 误差 {traj_err*1000:.1f}mm平均")
