"""
这是linux版本的测试与piper_X机械臂连接是否成功返回关节角度及其夹爪的开合
使用piper_sdk
首先需要激活CAN0:
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

ip -details link show can0
"""

import time
import numpy as np
from piper_sdk import C_PiperInterface


def joint_to_rad(msg):
    return np.array([
        msg.joint_1,
        msg.joint_2,
        msg.joint_3,
        msg.joint_4,
        msg.joint_5,
        msg.joint_6,
    ], dtype=np.float32) * 0.001 * np.pi / 180.0


piper = C_PiperInterface(can_name="can0")
piper.ConnectPort()

time.sleep(1.0)

while True:

    # -------------------------
    # Leader：目标命令
    # -------------------------
    leader = piper.GetArmJointCtrl()
    leader_gripper = piper.GetArmGripperCtrl()

    action_arm = joint_to_rad(leader.joint_ctrl)

    action_gripper = (
        leader_gripper.gripper_ctrl.grippers_angle * 1e-6
    )

    # -------------------------
    # Follower：实际反馈
    # -------------------------
    follower = piper.GetArmJointMsgs()
    follower_gripper = piper.GetArmGripperMsgs()

    qpos_arm = joint_to_rad(follower.joint_state)

    qpos_gripper = (
        follower_gripper.gripper_state.grippers_angle * 1e-6
    )

    print()
    print("ACTION :", np.round(action_arm, 3))
    print("QPOS   :", np.round(qpos_arm, 3))
    print(
        "GRIPPER:",
        round(action_gripper, 4),
        round(qpos_gripper, 4)
    )

    time.sleep(0.05)