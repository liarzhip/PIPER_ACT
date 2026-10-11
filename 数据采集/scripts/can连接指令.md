### can激活linux指令
```sh
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up

ip -details link show can0
```