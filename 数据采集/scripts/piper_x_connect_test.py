"""
这是windows版本的测试与piper_X机械臂连接并通过can获取机械臂信息
直接使用pyAgxArm
"""

import time
from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    PiperFW,
    create_agx_arm_config,
)

cfg = create_agx_arm_config(
    robot=ArmModel.PIPER_X,
    firmeware_version=PiperFW.DEFAULT,
    interface="agx_cando",
    channel="0",
)

robot = AgxArmFactory.create_arm(cfg)
print("Connecting PiPER-X...")
robot.connect()
print("Connected.")

try:
    while True:
        joint = robot.get_joint_angles()
        if joint is not None:
            print(
                "qpos:",
                joint.msg,
                "Hz:",
                joint.hz,
            )
        else:
            print("No joint feedback")
        time.sleep(0.05)
finally:
    robot.disconnect()