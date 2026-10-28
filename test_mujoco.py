"""
测试 MuJoCo 环境能不能跑起来。
加载 scene.xml，跑 100 步物理仿真，打印机械臂末端位置。
"""
import mujoco
import numpy as np

# 加载场景
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

print(f"模型加载成功")
print(f"自由度（qpos）数量：{model.nq}")
print(f"控制（ctrl）数量：{model.nu}")

# 跑 100 步
print("\n开始仿真 100 步...")
for i in range(100):
    mujoco.mj_step(model, data)

# 打印关节角度和末端位置
print(f"\n仿真结束：")
print(f"关节角度（qpos）：{data.qpos[:]}")
print(f"时间：{data.time:.3f} 秒")

# 找机械臂末端的 body id
try:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee")
    if ee_id >= 0:
        ee_pos = data.xpos[ee_id]
        print(f"末端位置：({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})")
    else:
        print("（没找到 ee body，用默认 body 0 的位置代替）")
        print(f"body 0 位置：{data.xpos[0]}")
except Exception as e:
    print(f"读取末端位置出错：{e}")

print("\n✅ MuJoCo 环境跑通！")
