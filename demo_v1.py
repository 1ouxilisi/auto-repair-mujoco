"""
auto-repair-mujoco V1 demo：多步维修任务 + 轨迹预测

任务：4步完整维修动作
  1. 从 home 移动到电池包上方
  2. 下降到电池包表面
  3. 拿起绝缘扳手
  4. 举到安全位置

模型：Transformer decoder 预测整条动作轨迹（不是单步预测）
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 非交互后端，不需要弹窗
import matplotlib.pyplot as plt

# ============ 1. 机械臂正运动学（2-link 平面） ============
L1, L2 = 0.3, 0.25  # 连杆长度（米）

def forward_kinematics(theta):
    """theta: (..., 2) → (..., 2) 末端位置"""
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# ============ 2. 多步任务关键帧（4个阶段） ============
# 阶段1：home → 电池包上方
# 阶段2：电池包上方 → 电池包表面（下降）
# 阶段3：电池包表面 → 拿起扳手位置
# 阶段4：扳手位置 → 安全位置（举起）

# 每个阶段的关节角度关键帧
STAGES = [
    # (阶段名, 起始关节角, 结束关节角)
    ("移动到电池包上方", [0.5, 0.5], [-0.5, 0.5]),
    ("下降到表面",       [-0.5, 0.5], [-0.5, 1.2]),
    ("移动到扳手位置",   [-0.5, 1.2], [-0.2, 1.0]),
    ("举起扳手到安全位", [-0.2, 1.0], [0.2, 0.6]),
]

def generate_demo_trajectory(stages, steps_per_stage=25):
    """生成完整演示轨迹"""
    traj = []
    for stage_name, start, end in stages:
        start = np.array(start)
        end = np.array(end)
        for s in range(steps_per_stage):
            alpha = s / steps_per_stage
            theta = (1 - alpha) * start + alpha * end
            traj.append(theta)
    traj.append(np.array(stages[-1][2]))
    return np.array(traj)

# ============ 3. 生成数据 ============
print("=" * 60)
print("[V1] 多步维修任务 + 轨迹预测 Demo")
print("=" * 60)

demo_traj = generate_demo_trajectory(STAGES, steps_per_stage=25)
print(f"演示轨迹长度：{len(demo_traj)} 步")
print(f"任务阶段：{len(STAGES)} 步（移动→下降→拿扳手→举起）")

# 观测：末端位置 + 当前阶段编号
obs_list = []
act_list = []

for i, theta in enumerate(demo_traj):
    pos = forward_kinematics(theta)
    # 观测 = 末端位置 + 阶段 one-hot（前 30% 步是阶段1，以此类推）
    stage = min(i * len(STAGES) // len(demo_traj), len(STAGES) - 1)
    obs = np.concatenate([pos, np.eye(len(STAGES))[stage]])
    obs_list.append(obs)
    act_list.append(theta)

obs = np.array(obs_list)  # (T, 2+4=6)
act = np.array(act_list)  # (T, 2)

# 加噪声
np.random.seed(42)
obs += np.random.normal(0, 0.003, obs.shape)
act += np.random.normal(0, 0.003, act.shape)

print(f"obs 维度：{obs.shape[1]}（2位置 + {len(STAGES)}阶段 one-hot）")
print(f"act 维度：{act.shape[1]}（2关节角度）")

# ============ 4. 序列预测模型：Transformer Decoder ============
print("\n" + "=" * 60)
print("[1/3] 训练序列预测模型")
print("=" * 60)

device = 'cpu'

# 标准化
obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
act_t = torch.tensor(act, dtype=torch.float32).to(device)

obs_mean, obs_std = obs_t.mean(0), obs_t.std(0) + 1e-6
act_mean, act_std = act_t.mean(0), act_t.std(0) + 1e-6

obs_n = (obs_t - obs_mean) / obs_std
act_n = (act_t - act_mean) / act_std

class TrajectoryTransformer(nn.Module):
    """简化版 Transformer：根据起始观测，预测整条动作轨迹"""
    def __init__(self, obs_dim=6, act_dim=2, seq_len=101, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.obs_proj = nn.Linear(obs_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.act_head = nn.Linear(d_model, act_dim)
        
        # 用一个可学习的 query token 作为"起始提示"
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
    def forward(self, start_obs):
        """start_obs: (B, obs_dim) → 预测整条轨迹 (B, seq_len, act_dim)"""
        B = start_obs.shape[0]
        
        # 把起始观测投影成 memory
        memory = self.obs_proj(start_obs).unsqueeze(1)  # (B, 1, d_model)
        
        # 生成 query：序列长度个 token
        query = self.pos_embed.expand(B, -1, -1)  # (B, seq_len, d_model)
        
        # Transformer decoder
        out = self.decoder(query, memory)  # (B, seq_len, d_model)
        
        # 预测动作
        act_pred = self.act_head(out)  # (B, seq_len, act_dim)
        return act_pred

model = TrajectoryTransformer(obs_dim=6, act_dim=2, seq_len=len(demo_traj)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 训练：只用第一个观测作为输入，预测整条轨迹
start_obs_n = obs_n[0:1]  # (1, 6)
target_act_n = act_n.unsqueeze(0)  # (1, T, 2)

epochs = 800
for ep in range(epochs):
    pred_act_n = model(start_obs_n)
    loss = loss_fn(pred_act_n, target_act_n)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 100 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

# ============ 5. Rollout：验证轨迹预测 ============
print("\n" + "=" * 60)
print("[2/3] Rollout 验证")
print("=" * 60)

model.eval()
with torch.no_grad():
    pred_act_n = model(start_obs_n)  # (1, T, 2)
    pred_act = pred_act_n * act_std + act_mean
    pred_act = pred_act.cpu().numpy()[0]  # (T, 2)

# 计算末端轨迹
pred_pos = np.array([forward_kinematics(theta) for theta in pred_act])
true_pos = np.array([forward_kinematics(theta) for theta in demo_traj])

# 最终位置误差
final_err = np.linalg.norm(pred_pos[-1] - true_pos[-1])
# 整条轨迹的平均误差
traj_err = np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))

print(f"轨迹长度：{len(pred_pos)} 步")
print(f"最终位置误差：{final_err*1000:.1f} mm")
print(f"整条轨迹平均误差：{traj_err*1000:.1f} mm")

# ============ 6. 可视化：画出轨迹 ============
print("\n" + "=" * 60)
print("[3/3] 生成轨迹图")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：机械臂末端轨迹
ax1.plot(true_pos[:, 0], true_pos[:, 1], 'b-o', label='真实演示', markersize=3)
ax1.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--x', label='预测轨迹', markersize=3)
ax1.scatter([0], [0], c='k', s=100, marker='s', label='基座')
ax1.scatter(true_pos[-1, 0], true_pos[-1, 1], c='g', s=200, marker='*', label='目标点')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('机械臂末端轨迹')
ax1.legend()
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 右图：关节角度随时间变化
ax2.plot(act[:, 0], 'b-', label='关节1（真实）', alpha=0.7)
ax2.plot(act[:, 1], 'g-', label='关节2（真实）', alpha=0.7)
ax2.plot(pred_act[:, 0], 'b--', label='关节1（预测）')
ax2.plot(pred_act[:, 1], 'g--', label='关节2（预测）')
ax2.set_xlabel('时间步')
ax2.set_ylabel('关节角度 (rad)')
ax2.set_title('关节角度轨迹')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v1_trajectory.png', dpi=150, bbox_inches='tight')
print("轨迹图已保存：v1_trajectory.png")

# ============ 7. 结果总结 ============
print("\n" + "=" * 60)
print("V1 Demo 完成！")
print("=" * 60)
print(f"✅ 多步任务：{len(STAGES)} 个阶段，{len(demo_traj)} 步完整动作")
print(f"✅ 序列预测：Transformer decoder 从起始观测预测整条轨迹")
print(f"✅ 最终位置误差：{final_err*1000:.1f} mm")
print(f"✅ 轨迹图：v1_trajectory.png")
print()
print("下一步路线：")
print("  V1.1：加更多噪声，测试鲁棒性")
print("  V1.2：换成 Diffusion Policy，复现 PushT 管线")
print("  V2：加多步骤维修任务（断电→验电→拆护板→...）")
