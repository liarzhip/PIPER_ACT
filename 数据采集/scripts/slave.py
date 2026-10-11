from piper_sdk import C_PiperInterface

piper = C_PiperInterface(can_name="can0")
piper.ConnectPort()

# 0xFC = motion output arm
piper.MasterSlaveConfig(0xFC, 0, 0, 0)

print("Follower configured")