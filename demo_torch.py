"""
auto-repair-mujoco 纯 PyTorch 版 demo（不依赖 mujoco）
用 2-link 平面机械臂的正运动学演示：采集演示数据 → 训练 MLP → rollout 验证。

这是一个最小可行 demo，证明整个 pipeline 跑通。
后面换成真 MuJoCo 环境时，只需要替换"环境"那一层。
"""
import numpy as np
import torch
import torch.nn as nn

# ============ 1. 定义 2-link 机械臂正运动学 ============
L1, L2 = 0.3, 0.25  # 连杆长度（米）

def forward_kinematics(theta):
    """theta: (..., 2) 关节角度 → (..., 2) 末端位置"""
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

# ============ 2. 预设关键帧：从 home 到电池包上方 ============
# 关节角度（弧度）
KEYFRAMES = np.array([
    [0.5, 0.5],    # home：手臂收在身前
    [0.3, 0.8],    # 抬起
    [0.0, 1.0],    # 转向目标方向
    [-0.3, 0.8],   # 伸出去
    [-0.5, 0.5],   # 到达目标上方
    [-0.5, 0.5],   # 保持
])

def interpolate(keyframes, steps_per_kf=30):
    traj = []
    for i in range(len(keyframes) - 1):
        for s in range(steps_per_kf):
            alpha = s / steps_per_kf
            traj.append((1 - alpha) * keyframes[i] + alpha * keyframes[i + 1])
    traj.append(keyframes[-1])
    return np.array(traj)

# ============ 3. 采集演示数据 ============
print("=" * 50)
print("[1/4] 采集演示数据")
print("=" * 50)

traj = interpolate(KEYFRAMES, steps_per_kf=30)
target_pos = forward_kinematics(KEYFRAMES[-1])  # 目标位置

obs_list = []  # 末端位置
act_list = []  # 关节角度

for theta in traj:
    pos = forward_kinematics(theta)
    obs_list.append(pos)
    act_list.append(theta)

obs = np.array(obs_list)  # (T, 2)
act = np.array(act_list)  # (T, 2)

# 加噪声，模拟真实演示的不完美
np.random.seed(42)
obs += np.random.normal(0, 0.005, obs.shape)
act += np.random.normal(0, 0.005, act.shape)

print(f"轨迹长度：{len(traj)} 步")
print(f"目标位置（电池包上方）：({target_pos[0]:.3f}, {target_pos[1]:.3f})")
print(f"数据形状：obs={obs.shape}, act={act.shape}")

# ============ 4. 训练 MLP：末端位置 → 关节角度 ============
print("\n" + "=" * 50)
print("[2/4] 训练 MLP 策略")
print("=" * 50)

device = 'cpu'
obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
act_t = torch.tensor(act, dtype=torch.float32).to(device)

# 标准化
obs_mean, obs_std = obs_t.mean(0), obs_t.std(0) + 1e-6
act_mean, act_std = act_t.mean(0), act_t.std(0) + 1e-6
obs_n = (obs_t - obs_mean) / obs_std
act_n = (act_t - act_mean) / act_std

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 500
for ep in range(epochs):
    pred = model(obs_n)
    loss = loss_fn(pred, act_n)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 100 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

# ============ 5. Rollout：验证策略能不能到达目标 ============
print("\n" + "=" * 50)
print("[3/4] Rollout 验证")
print("=" * 50)

model.eval()
target_t = torch.tensor(target_pos, dtype=torch.float32).to(device)
target_n = (target_t - obs_mean) / obs_std

with torch.no_grad():
    pred_act_n = model(target_n.unsqueeze(0))
    pred_act = pred_act_n * act_std + act_mean

pred_theta = pred_act.cpu().numpy()[0]
final_pos = forward_kinematics(pred_theta)
dist = np.linalg.norm(final_pos - target_pos)

print(f"目标位置：({target_pos[0]:.3f}, {target_pos[1]:.3f})")
print(f"预测关节角度：({pred_theta[0]:.3f}, {pred_theta[1]:.3f})")
print(f"预测末端位置：({final_pos[0]:.3f}, {final_pos[1]:.3f})")
print(f"到目标距离：{dist:.4f} 米")

# ============ 6. 结果 ============
print("\n" + "=" * 50)
print("[4/4] 结果")
print("=" * 50)

if dist < 0.02:
    print(f"✅ 成功！机械臂到达电池包上方，误差 {dist*1000:.1f} mm")
    print("   pipeline 跑通：采集数据 → MLP 行为克隆 → 到达目标")
    print("   下一步：换成真 MuJoCo 环境，加多步骤任务（断电→验电→拆护板）")
else:
    print(f"⚠️ 距离 {dist:.4f} 米，需要更多训练或更多演示数据")

print("\nDemo 完成。")
