import asyncio
from bleak import BleakClient, BleakScanner
import struct
import time
import socket

client_socket = None  # 用于存储 TCP 客户端连接
receiveCount = 0  # 用于记录接收到的数据包数量
before_time = 0  # 用于记录上次计算采样率的时间

# 定期任务：每10秒计算并打印采样率
async def periodic_task():
    global receiveCount, before_time
    while True:
        # 计算自上次记录以来的时间差
        res_time = time.time() - before_time
        # 计算并四舍五入采样率（接收的数据包数除以时间差）
        hz = int(receiveCount / res_time + 0.5)
        print(f"采样ui率: {hz} 时间: {res_time}")
        before_time = time.time()  # 更新记录时间
        receiveCount = 0  # 重置数据包计数
        await asyncio.sleep(10)  # 等待10秒
  
# 扫描并查找目标设备的MAC地址
async def findMac():
    devices = await BleakScanner.discover()  # 扫描设备  
    count = 0
    while 1:
        print(f"扫描第{count + 1}轮")
        for d in devices:  # d为设备信息，包含d.name为设备名称，d.address为设备地址
            if __name__ == '__main__':
                if d.name == "Motren_E":  # 查找目标设备名称为"Motren_E"
                    print(f"发现设备Motren_E, MAC地址为: {d.address}")
                    return d  # 返回目标设备信息
        await asyncio.sleep(1)  # 每次扫描间隔1秒

# 解析接收到的 IMU 数a
# 据（二进制格式）
def parse_imu_data(data):
    # 解析6个16位的有符号整数，分别对应陀螺仪和加速度传+-感器数据
    imu_gyro_x, imu_gyro_y, imu_gyro_z, imu_accel_x, imu_accel_y, imu_accel_z = struct.unpack('<hhhhhh', data)
    # 对陀螺仪数据按比例缩放（单位：degree/s）
    data_gyro = (imu_gyro_x, imu_gyro_y, imu_gyro_z)
    data_gyro = [data / 131 for data in data_gyro]
    # 对加速度计数据按比例缩放
    data_accel = (imu_accel_x, imu_accel_y, imu_accel_z)
    data_accel = [data / 16384 for data in data_accel]
    return data_accel

# 通知回调函数，接收并处理IMU数据
def notification_handler(sender, data):
    data = data[1:]  # 去掉数据头，获取有效IMU数据
    data = parse_imu_data(data)  # 解析IMU数据
    global receiveCount
    receiveCount += 1  # 增加接收到的数据包计数
    global client_socket
    # 组装消息字符串，并发送到TCP客户端
    meg = f"{0},{data[0]},{data[1]},{data[2]}\n"
    client_socket.send(meg.encode("utf-8"))  # 将消息编码并通过Socket发送

# 主任务：连接设备并处理IMU数据通知
async def main(address):
    print("connecting to device...")
    print(f"尝试连接 {address}")
    par_notification_characteristic = "fec26ec4-6d71-4442-9f81-55bc21d658d6"  # 设备的通知特征UUID
    global before_time
    # 使用BleakClient连接设备
    async with BleakClient(address) as client:
        before_time = time.time()  # 记录开始时间
        # 启动定期任务，用于计算和显示采样率
        periodic_task_handle = asyncio.create_task(periodic_task())
        # 开始监听设备的通知特征，收到通知时调用notification_handler
        await client.start_notify(par_notification_characteristic, notification_handler)
        # 运行5000秒后自动断开连接                            
        await asyncio.sleep(5000.0)
        await client.stop_notify(par_notification_characteristic)  # 停止通知
        periodic_task_handle.cancel()  # 取消定期任务 bhmmmm.g

# 程序入口，初始化并运行
if __name__ == "__main__": 
    d = asyncio.run(findMac())  # 扫描并查找目标设备
    address = d.address  # 获取目标设备的MAC地址
    hostName = socket.gethostname()  # 获取主机名
    local_ip = socket.gethostbyname(hostName)  # 获取本机IP地址
    print(f"本地 IP 地址: {local_ip}")
    # 创建TCP客户端Socket，AF_INET表示使用IPv4，SOCK_STREAM表示使用TCP协议
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((hostName, 12345))  # 连接到指定端口的服务器
    print(f"服务端启动成功，在 {12345} 端口等待客户端连接...")
    asyncio.run(main(address))  # 运行主任务，连接蓝牙设备并处理IMU数据
   