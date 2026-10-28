"""
auto-repair-mujoco V3 demo：Diffusion Policy 简化版

对比：
  - Baseline: Transformer decoder（V2 用的）
  - Diffusion Policy: 加噪-去噪生成轨迹

任务：8阶段完整电池包检查流程（同 V2）
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ 1. 机械臂 + 任务定义（同 V2） ============
L1, L2 = 0.3, 0.25

def forward_kinematics(theta):
    t1, t2 = theta[..., 0], theta[..., 1]
    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return np.stack([x, y], axis=-1)

STAGES = [
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

# ============ 2. 生成数据 ============
print("=" * 60)
print("[V3] Diffusion Policy vs Transformer Baseline")
print("=" * 60)

demo_traj = generate_trajectory(STAGES, steps_per_stage=15)
T = len(demo_traj)
act_dim = 2

# 生成多条带噪声的演示轨迹（模拟多个演示者）
np.random.seed(42)
n_demos = 5
all_trajs = []
for _ in range(n_demos):
    noise = np.random.normal(0, 0.005, demo_traj.shape)
    all_trajs.append(demo_traj + noise)
all_trajs = np.array(all_trajs)  # (n_demos, T, act_dim)

print(f"演示轨迹数：{n_demos} 条")
print(f"每条轨迹长度：{T} 步")
print(f"动作维度：{act_dim}（2关节角度）")

# 标准化
act_mean = all_trajs.mean(axis=(0, 1))
act_std = all_trajs.std(axis=(0, 1)) + 1e-6
all_trajs_n = (all_trajs - act_mean) / act_std

# 起始观测（每条轨迹的第一个动作）
start_act = all_trajs_n[:, 0:1, :]  # (n_demos, 1, act_dim)

# ============ 3. Baseline: Transformer ============
print("\n" + "=" * 60)
print("[1/4] 训练 Baseline Transformer")
print("=" * 60)

device = 'cpu'

class TransformerBaseline(nn.Module):
    def __init__(self, act_dim=2, seq_len=121, d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.start_proj = nn.Linear(act_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.act_head = nn.Linear(d_model, act_dim)
        
    def forward(self, start_act):
        B = start_act.shape[0]
        memory = self.start_proj(start_act)
        query = self.pos_embed.expand(B, -1, -1)
        out = self.decoder(query, memory)
        return self.act_head(out)

baseline = TransformerBaseline(act_dim=act_dim, seq_len=T).to(device)
opt_b = torch.optim.Adam(baseline.parameters(), lr=5e-4)
loss_fn = nn.MSELoss()

target_t = torch.tensor(all_trajs_n, dtype=torch.float32).to(device)
start_t = torch.tensor(start_act, dtype=torch.float32).to(device)

epochs = 400
for ep in range(epochs):
    pred = baseline(start_t)
    loss = loss_fn(pred, target_t)
    opt_b.zero_grad()
    loss.backward()
    opt_b.step()
    if (ep + 1) % 100 == 0:
        print(f"  epoch {ep+1}/{epochs}  loss={loss.item():.6f}")

# ============ 4. Diffusion Policy 模型 ============
print("\n" + "=" * 60)
print("[2/4] 训练 Diffusion Policy")
print("=" * 60)

class DiffusionPolicy(nn.Module):
    """简化版 Diffusion Policy：预测噪声，逐步去噪生成轨迹"""
    def __init__(self, act_dim=2, seq_len=121, d_model=64, nhead=4, num_layers=2, n_diffusion_steps=10):
        super().__init__()
        self.seq_len = seq_len
        self.n_diffusion_steps = n_diffusion_steps
        
        # 输入：带噪声的轨迹 + 时间步编码 + 起始观测
        self.act_proj = nn.Linear(act_dim, d_model)
        self.time_embed = nn.Linear(1, d_model)
        self.start_proj = nn.Linear(act_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # 用 Transformer encoder 来去噪
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.noise_pred = nn.Linear(d_model, act_dim)
        
    def forward(self, noisy_traj, t, start_act):
        """
        noisy_traj: (B, T, act_dim) 带噪声的轨迹
        t: (B, 1) 时间步（0~1）
        start_act: (B, 1, act_dim) 起始观测
        """
        B = noisy_traj.shape[0]
        
        # 投影各部分
        act_emb = self.act_proj(noisy_traj)  # (B, T, d_model)
        time_emb = self.time_embed(t).unsqueeze(1)  # (B, 1, d_model)
        start_emb = self.start_proj(start_act)  # (B, 1, d_model)
        
        # 加位置编码
        x = act_emb + self.pos_embed.expand(B, -1, -1)
        
        # 加上时间步和起始观测
        x = x + time_emb + start_emb
        
        # Transformer encoder
        h = self.encoder(x)
        
        # 预测噪声
        noise_pred = self.noise_pred(h)
        return noise_pred
    
    def generate(self, start_act, n_steps=None):
        """从随机噪声开始，逐步去噪生成轨迹"""
        if n_steps is None:
            n_steps = self.n_diffusion_steps
        
        B = start_act.shape[0]
        device = start_act.device
        
        # 从纯噪声开始
        x = torch.randn(B, self.seq_len, act_dim, device=device)
        
        # 逐步去噪
        for i in range(n_steps):
            t = torch.ones(B, 1, device=device) * (1 - i / n_steps)
            noise_pred = self.forward(x, t, start_act)
            # 简单的去噪步长
            alpha = 1.0 / (n_steps - i)
            x = x - alpha * noise_pred
        
        return x

diffusion = DiffusionPolicy(act_dim=act_dim, seq_len=T, n_diffusion_steps=10).to(device)
opt_d = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

# 训练：随机采样时间步，加噪声，预测噪声
diffusion_epochs = 500
for ep in range(diffusion_epochs):
    # 随机采样一批轨迹
    batch_idx = np.random.randint(0, n_demos, size=2)
    batch = target_t[batch_idx]  # (2, T, act_dim)
    batch_start = start_t[batch_idx]
    
    # 随机采样时间步
    t = torch.rand(2, 1, device=device)  # (B, 1)
    
    # 加噪声（噪声强度和时间步成正比）
    noise = torch.randn_like(batch)
    noisy_batch = batch + t.unsqueeze(-1) * noise  # (B,1,1) * (B,T,act_dim)
    
    # 预测噪声
    noise_pred = diffusion(noisy_batch, t, batch_start)
    loss = loss_fn(noise_pred, noise)
    
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    
    if (ep + 1) % 100 == 0:
        print(f"  epoch {ep+1}/{diffusion_epochs}  loss={loss.item():.6f}")

# ============ 5. 对比验证 ============
print("\n" + "=" * 60)
print("[3/4] 对比两种方法的效果")
print("=" * 60)

baseline.eval()
diffusion.eval()

with torch.no_grad():
    # Baseline 预测
    baseline_pred = baseline(start_t)  # (n_demos, T, act_dim)
    
    # Diffusion 生成
    diffusion_pred = diffusion.generate(start_t, n_steps=20)  # 多去噪几步更精确

# 反标准化
baseline_pred_np = baseline_pred.cpu().numpy() * act_std + act_mean
diffusion_pred_np = diffusion_pred.cpu().numpy() * act_std + act_mean
true_np = all_trajs * 1.0  # 已经是真实值

# 计算误差
def compute_errors(pred, true):
    """pred, true: (n_demos, T, act_dim)"""
    # 每个 demo 的最终位置误差
    final_errs = []
    for i in range(len(pred)):
        pred_pos = np.array([forward_kinematics(theta) for theta in pred[i]])
        true_pos = np.array([forward_kinematics(theta) for theta in true[i]])
        final_err = np.linalg.norm(pred_pos[-1] - true_pos[-1])
        final_errs.append(final_err)
    
    # 平均轨迹误差
    traj_errs = []
    for i in range(len(pred)):
        pred_pos = np.array([forward_kinematics(theta) for theta in pred[i]])
        true_pos = np.array([forward_kinematics(theta) for theta in true[i]])
        traj_err = np.mean(np.linalg.norm(pred_pos - true_pos, axis=1))
        traj_errs.append(traj_err)
    
    return np.mean(final_errs), np.mean(traj_errs)

baseline_final, baseline_traj = compute_errors(baseline_pred_np, true_np)
diffusion_final, diffusion_traj = compute_errors(diffusion_pred_np, true_np)

print(f"Baseline Transformer:")
print(f"  最终位置误差：{baseline_final*1000:.1f} mm")
print(f"  平均轨迹误差：{baseline_traj*1000:.1f} mm")
print()
print(f"Diffusion Policy:")
print(f"  最终位置误差：{diffusion_final*1000:.1f} mm")
print(f"  平均轨迹误差：{diffusion_traj*1000:.1f} mm")

# ============ 6. 可视化对比 ============
print("\n" + "=" * 60)
print("[4/4] 生成对比图")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 画第一条演示的对比
demo_idx = 0
true_traj = true_np[demo_idx]
baseline_traj = baseline_pred_np[demo_idx]
diffusion_traj = diffusion_pred_np[demo_idx]

true_pos = np.array([forward_kinematics(theta) for theta in true_traj])
baseline_pos = np.array([forward_kinematics(theta) for theta in baseline_traj])
diffusion_pos = np.array([forward_kinematics(theta) for theta in diffusion_traj])

# 左图：真实轨迹 vs Baseline
ax = axes[0]
ax.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True', linewidth=2)
ax.plot(baseline_pos[:, 0], baseline_pos[:, 1], 'r--', label='Transformer', linewidth=2)
ax.scatter([0], [0], c='k', s=100, marker='s')
ax.set_title(f'Baseline Transformer\nFinal error: {baseline_final*1000:.1f}mm')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 中图：真实轨迹 vs Diffusion
ax = axes[1]
ax.plot(true_pos[:, 0], true_pos[:, 1], 'b-', label='True', linewidth=2)
ax.plot(diffusion_pos[:, 0], diffusion_pos[:, 1], 'g--', label='Diffusion Policy', linewidth=2)
ax.scatter([0], [0], c='k', s=100, marker='s')
ax.set_title(f'Diffusion Policy\nFinal error: {diffusion_final*1000:.1f}mm')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 右图：误差对比柱状图
ax = axes[2]
methods = ['Transformer', 'Diffusion']
final_errs = [baseline_final*1000, diffusion_final*1000]
traj_errs = [baseline_traj*1000, diffusion_traj*1000]

x = np.arange(len(methods))
width = 0.35
ax.bar(x - width/2, final_errs, width, label='Final Error (mm)', color='#ff6b6b')
ax.bar(x + width/2, traj_errs, width, label='Avg Traj Error (mm)', color='#4ecdc4', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel('Error (mm)')
ax.set_title('Error Comparison')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v3_diffusion_comparison.png', dpi=150, bbox_inches='tight')
print("对比图已保存：v3_diffusion_comparison.png")

# ============ 7. 总结 ============
print("\n" + "=" * 60)
print("V3 Demo 完成！")
print("=" * 60)
print(f"✅ 8阶段完整维修流程（同 V2）")
print(f"✅ Baseline: Transformer decoder")
print(f"✅ Diffusion Policy: 加噪-去噪生成")
print(f"✅ 对比实验：两种方法在 10 条演示轨迹上的表现")
print(f"✅ 对比可视化：v3_diffusion_comparison.png")
print()
print("下一步路线：")
print("  V3.1：加更多噪声水平测试，看鲁棒性差异")
print("  V3.2：换成真 MuJoCo 环境")
print("  V4：打包成技能包，上架机器人应用商店")
