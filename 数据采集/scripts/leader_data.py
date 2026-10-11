import time
from piper_sdk import C_PiperInterface_V2

piper = C_PiperInterface_V2("can0")
piper.ConnectPort()

while True:
    joint_ctrl = piper.GetArmJointCtrl()
    gripper_ctrl = piper.GetArmGripperCtrl()

    print("Joint Ctrl Hz:", joint_ctrl.Hz)
    print("Gripper Ctrl Hz:", gripper_ctrl.Hz)
    print(joint_ctrl)

    time.sleep(0.1)