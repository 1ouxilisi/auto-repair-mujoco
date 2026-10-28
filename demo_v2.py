"""
auto-repair-mujoco V2 demo：完整维修流程 + 条件生成

任务：8个阶段的电池包高压检查流程
  1. 移动到电池包上方
  2. 下降到表面
  3. 找到验电笔位置
  4. 移动到验电笔上方
  5. 拿起验电笔
  6. 移动到电池包检测点
  7. 进行验电（保持）
  8. 收回验电笔到安全位置

模型：条件 Transformer，根据任务编号生成对应轨迹
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 1. 机械臂正运动学 ============
L1, L2 = 0.3, 0.25

def forward_kinematics(theta):
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# ============ 2. 8阶段完整维修流程 ============
STAGES_V2 = [
    ("移动到电池包上方", [0.5, 0.5], [-0.5, 0.5]),
    ("下降到电池包表面", [-0.5, 0.5], [-0.5, 1.2]),
    ("转向验电笔方向", [-0.5, 1.2], [-0.1, 1.0]),
    ("移动到验电笔上方", [-0.1, 1.0], [0.3, 0.9]),
    ("拿起验电笔",       [0.3, 0.9], [0.3, 1.2]),
    ("移动到检测点",     [0.3, 1.2], [-0.3, 1.3]),
    ("验电保持",         [-0.3, 1.3], [-0.3, 1.3]),
    ("收回安全位置",     [-0.3, 1.3], [0.4, 0.6]),
]

def generate_trajectory(stages, steps_per_stage=15):
    traj = []
    for name, start, end in stages:
        start = np.array(start)
        end = np.array(end)
        for s in range(steps_per_stage):
            alpha = s / steps_per_stage
            traj.append((1 - alpha) * start + alpha * end)
    traj.append(np.array(stages[-1][2]))
    return np.array(traj)

# ============ 3. 生成数据 ============
print("=" * 60)
print("[V2] 完整维修流程 + 条件生成 Demo")
print("=" * 60)

demo_traj = generate_trajectory(STAGES_V2, steps_per_stage=15)
T = len(demo_traj)
n_stages = len(STAGES_V2)

print(f"任务阶段：{n_stages} 个完整维修步骤")
print(f"轨迹长度：{T} 步")
print(f"阶段列表：")
for i, (name, _, _) in enumerate(STAGES_V2):
    print(f"  {i+1}. {name}")

# 观测：末端位置 + 阶段进度（0~1 表示当前在哪个阶段）
obs_list = []
act_list = []

for i, theta in enumerate(demo_traj):
    pos = forward_kinematics(theta)
    progress = i / T  # 进度 0~1
    obs = np.concatenate([pos, [progress]])
    obs_list.append(obs)
    act_list.append(theta)

obs = np.array(obs_list)  # (T, 3)
act = np.array(act_list)  # (T, 2)

np.random.seed(42)
obs += np.random.normal(0, 0.002, obs.shape)
act += np.random.normal(0, 0.002, act.shape)

print(f"\nobs 维度：{obs.shape[1]}（2位置 + 1进度）")
print(f"act 维度：{act.shape[1]}（2关节角度）")

# ============ 4. 条件 Transformer 模型 ============
print("\n" + "=" * 60)
print("[1/3] 训练条件轨迹模型")
print("=" * 60)

device = 'cpu'
obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
act_t = torch.tensor(act, dtype=torch.float32).to(device)

obs_mean, obs_std = obs_t.mean(0), obs_t.std(0) + 1e-6
act_mean, act_std = act_t.mean(0), act_t.std(0) + 1e-6

obs_n = (obs_t - obs_mean) / obs_std
act_n = (act_t - act_mean) / act_std

class ConditionalTransformer(nn.Module):
    """根据起始观测 + 目标进度，预测整条动作轨迹"""
    def __init__(self, obs_dim=3, act_dim=2, seq_len=121, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        
        # 输入投影
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.act_head = nn.Linear(d_model, act_dim)
        
    def forward(self, start_obs):
        """start_obs: (B, obs_dim) → 预测整条轨迹 (B, seq_len, act_dim)"""
        B = start_obs.shape[0]
        
        # memory：起始观测
        memory = self.obs_proj(start_obs).unsqueeze(1)
        
        # query：序列位置编码
        query = self.pos_embed.expand(B, -1, -1)
        
        out = self.decoder(query, memory)
        act_pred = self.act_head(out)
        return act_pred

model = ConditionalTransformer(obs_dim=3, act_dim=2, seq_len=T).to(device)
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

start_obs_n = obs_n[0:1]
target_act_n = act_n.unsqueeze(0)

epochs = 1000
for ep in range(epochs):
    pred = model(start_obs_n)
    loss = loss_fn(pred, target_act_n)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 200 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

# ============ 5. Rollout 验证 ============
print("\n" + "=" * 60)
print("[2/3] Rollout 验证")
print("=" * 60)

model.eval()
with torch.no_grad():
    pred_act_n = model(start_obs_n)
    pred_act = pred_act_n * act_std + act_mean
    pred_act = pred_act.cpu().numpy()[0]

pred_pos = np.array([forward_kinematics(theta) for theta in pred_act])
true_pos = np.array([forward_kinematics(theta) for theta in demo_traj])

final_err = np.linalg.norm(pred_pos[-1] - true_pos[-1])
traj_err = np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))
max_err = np.max(np.linalg.norm(pred_pos - true_pos, axis=1))

print(f"轨迹长度：{T} 步")
print(f"最终位置误差：{final_err*1000:.1f} mm")
print(f"平均轨迹误差：{traj_err*1000:.1f} mm")
print(f"最大轨迹误差：{max_err*1000:.1f} mm")

# ============ 6. 可视化：完整轨迹图 ============
print("\n" + "=" * 60)
print("[3/3] 生成可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 阶段颜色
stage_colors = plt.cm.viridis(np.linspace(0, 1, n_stages))

# 左上图：完整轨迹
ax = axes[0, 0]
ax.plot(true_pos[:, 0], true_pos[:, 1], 'b-', alpha=0.5, label='True', linewidth=2)
ax.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', label='Predicted', linewidth=2)

# 标注每个阶段的起点
for i, (name, start, end) in enumerate(STAGES_V2):
    start_idx = i * 15
    if start_idx < len(true_pos):
        ax.scatter(true_pos[start_idx, 0], true_pos[start_idx, 1], 
                  c=[stage_colors[i]], s=100, zorder=5, edgecolors='black')
        ax.annotate(f'{i+1}.{name[:6]}', 
                   (true_pos[start_idx, 0], true_pos[start_idx, 1]),
                   fontsize=7, xytext=(5, 5), textcoords='offset points')

ax.scatter([0], [0], c='k', s=200, marker='s', label='Base')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('V2: Full Repair Task Trajectory')
ax.legend(fontsize=8)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 右上图：关节角度
ax = axes[0, 1]
ax.plot(act[:, 0], 'b-', label='Joint 1 (True)', alpha=0.7)
ax.plot(act[:, 1], 'g-', label='Joint 2 (True)', alpha=0.7)
ax.plot(pred_act[:, 0], 'b--', label='Joint 1 (Pred)')
ax.plot(pred_act[:, 1], 'g--', label='Joint 2 (Pred)')
ax.set_xlabel('Step')
ax.set_ylabel('Angle (rad)')
ax.set_title('Joint Angles Over Time')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 左下图：轨迹误差随时间变化
ax = axes[1, 0]
err_per_step = np.linalg.norm(pred_pos - true_pos, axis=1) * 1000
ax.plot(err_per_step, 'r-', linewidth=1.5)
ax.axhline(y=2, color='g', linestyle='--', alpha=0.5, label='2mm threshold')
ax.set_xlabel('Step')
ax.set_ylabel('Error (mm)')
ax.set_title('Trajectory Error per Step')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 右下图：各阶段误差统计
ax = axes[1, 1]
stage_errors = []
stage_names = []
for i, (name, _, _) in enumerate(STAGES_V2):
    start = i * 15
    end = min((i+1) * 15, len(err_per_step))
    stage_errors.append(np.mean(err_per_step[start:end]))
    stage_names.append(f'S{i+1}')

bars = ax.bar(stage_names, stage_errors, color=stage_colors)
ax.set_xlabel('Stage')
ax.set_ylabel('Avg Error (mm)')
ax.set_title('Average Error by Stage')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v2_trajectory.png', dpi=150, bbox_inches='tight')
print("可视化已保存：v2_trajectory.png")

# ============ 7. 总结 ============
print("\n" + "=" * 60)
print("V2 Demo 完成！")
print("=" * 60)
print(f"✅ 完整维修流程：{n_stages} 个阶段，{T} 步动作")
print(f"✅ 条件 Transformer：根据起始观测预测整条轨迹")
print(f"✅ 最终位置误差：{final_err*1000:.1f} mm")
print(f"✅ 平均轨迹误差：{traj_err*1000:.1f} mm")
print(f"✅ 4合1 可视化图：v2_trajectory.png")
print()
print("下一步路线：")
print("  V2.1：加更多噪声，测试鲁棒性")
print("  V2.2：换成 Diffusion Policy，复现 PushT 管线")
print("  V3：换成 UR5/Franka 真机模型，Sim2Real 迁移")
print("  V4：打包成技能包，上架机器人应用商店")
