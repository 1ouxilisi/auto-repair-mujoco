"""
用训练好的策略在 MuJoCo 里跑 rollout，验证机械臂能不能复现"移动到电池包上方"。
打印每步末端位置和到目标的距离。

用法：
    python rollout.py --checkpoint checkpoints/bc_mlp.pt
"""
import argparse
import numpy as np
import torch
import mujoco
import os

class MLP(torch.nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, act_dim),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/bc_mlp.pt')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    model = mujoco.MjModel.from_xml_path(os.path.join(here, 'scene.xml'))
    data = mujoco.MjData(model)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    policy = MLP(ckpt['obs_dim'], ckpt['act_dim'])
    policy.load_state_dict(ckpt['model'])
    policy.eval()

    target_pos = data.site_xpos[model.site('target').id].copy()
    print(f"目标位置（电池包上方）：{target_pos}")

    # 重置
    data.qpos[:6] = [0.0, -0.5, 0.5, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    for step in range(args.steps):
        obs = np.concatenate([data.qpos[:6], data.site_xpos[model.site('ee_site').id]])
        obs_n = (obs - ckpt['obs_mean']) / ckpt['obs_std']
        with torch.no_grad():
            act_n = policy(torch.tensor(obs_n, dtype=torch.float32))
        act = act_n.numpy() * ckpt['act_std'] + ckpt['act_mean']
        data.ctrl[:6] = act
        mujoco.mj_step(model, data)

        ee = data.site_xpos[model.site('ee_site').id]
        dist = np.linalg.norm(ee - target_pos)
        if step % 20 == 0:
            print(f"step {step:3d}  ee=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})  到目标距离={dist:.4f}")

    ee_final = data.site_xpos[model.site('ee_site').id]
    final_dist = np.linalg.norm(ee_final - target_pos)
    print(f"\n最终到目标距离：{final_dist:.4f} 米")
    if final_dist < 0.1:
        print("✅ 成功！机械臂到达电池包上方。")
    else:
        print(f"⚠️ 还差 {final_dist:.3f} 米，继续训练或增加演示数据。")

if __name__ == '__main__':
    main()
