from piper_sdk import C_PiperInterface

piper = C_PiperInterface(can_name="can0")
piper.ConnectPort()

# 0xFA = teaching input arm
piper.MasterSlaveConfig(0xFA, 0, 0, 0)

print("Leader configured")