import socket
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import threading
import scipy
from datetime import datetime
import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel, firwin, freqz, lfilter

# 定义低通滤波器函数，用于过滤信号中的高频成分
def lowpass_filter(data, b, a):
    y = lfilter(b, a, data)  # 使用lfilter函数进行滤波,lfilter是一个直接II型结构的数字滤波器
    return y

# 初始化图形，绘制加速度数据
def initializeFigure(accresult, figsize, color_arr, dict_xyz):
    # accresult: 加速度数据矩阵
    # figsize: 图形大小
    # color_arr: 线条颜色
    # dict_xyz: 轴标签

    # 创建一个包含4个子图的图形窗口
    fig, axs = plt.subplots(4, 1, figsize=figsize, dpi=100)
    # plt.subplots()函数返回一个包含所有子图的numpy数组
    # plt.subplots(4, 1)表示创建一个包含4个子图的图形窗口，4行1列

    lines = []  # 存储绘制的线条对象
    # 遍历每个子图进行设置
    for i in range(4):
        if i == 0:
            # 修改后的第一条线条绘制，显示X轴的详细信息
            lines.append(
                axs[i].plot(detail_x_axis, accresult[1, -len(detail_x_axis):], color=color_arr[i], linewidth=1.5, alpha=1)[0])
            axs[i].set_title(dict_xyz[i], fontsize=16, loc="left", y=0.41, x=-0.12)

        else:
            # 绘制其余的X, Y, Z轴的数据
            lines.append(
                axs[i].plot(time_axis, accresult[i, :], color=color_arr[i], linewidth=1.5, alpha=0.7 if i != 3 else 1)[
                    0])
            axs[i].set_title(dict_xyz[i], fontsize=16, loc="left", y=0.41, x=-0.1)
        # axs[i].set_xticks([])  # 移除x轴刻度

        # 隐藏左右两边的坐标轴线
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].grid(axis="y", linestyle='--', linewidth=1, alpha=0.25)  # 为y轴添加网格线

        # 设置坐标轴线的样式和透明度
        axs[i].spines["bottom"].set_linestyle('--')
        axs[i].spines["bottom"].set_linewidth(1)
        axs[i].spines["bottom"].set_alpha(0.3)
        axs[i].spines["top"].set_linestyle('--')
        axs[i].spines["top"].set_linewidth(1)
        axs[i].spines["top"].set_alpha(0.3)

    # 绑定窗口关闭事件和键盘按键事件
    fig.canvas.mpl_connect('close_event', on_close) # 关闭窗口时调用on_close函数
    fig.canvas.mpl_connect('key_press_event', down_space_save_before) # 按下空格键时调用down_space_save_before函数
    return lines, axs, fig  # 返回绘制线条、子图和图形对象

# 获取当前时间，并格式化输出
def get_currentTime():
    current_time = datetime.now()  # 获取当前时间，return: YYYY-MM-DD HH:MM:SS.microsecond
    month = current_time.month
    day = current_time.day
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    return f"{month}-{day}-{hour}-{minute}-{second}"

# 处理接收到的数据，将其转换为数组形式
def process_data(message):
    message = message.decode("utf-8")  # 将字节数据解码为字符串
    if message == "":
        return None  # 如果消息为空，返回None
    arr = message.split("\n")[:-1]  # 去掉最后的空行
    arr = list(map(lambda x: x.split(","), arr))  # 按逗号分割每一行
    arr = np.array(arr, dtype=float)  # 转换为浮点数数组
    return arr

# 更新加速度结果数组，将新数据追加到旧数据后
def update_accresult(data, accresult):
    count = data.shape[0]  # 获取新数据的数量
    accresult = np.concatenate([accresult[:, count:], data.T], axis=1)  # 将新数据拼接到结果数组后
    return accresult

# 计算时间间隔，并格式化为分钟和秒
def get_TimeInterval(start_time, end_time) -> str:
    res_time = end_time - start_time
    minute = res_time // 60  # 计算分钟
    second = res_time % 60  # 计算秒
    minute = "{:.0f}".format(minute)
    second = "{:.2f}".format(second)
    if len(minute) == 1:
        minute = "0" + minute
    if len(second.split(".")[0]) == 1:
        second = "0" + second
    return f"{minute}:{second}"

# 更新绘图，刷新数据并调整Y轴范围
def update_figure(lines, axs, accresult):
    global detail_x_axis, spaceDown_time, timer_text
    for i, line in enumerate(lines):
# 修改后的更新逻辑，将Y轴改为X轴
        if i == 0:
            line.set_ydata(accresult[1, -len(detail_x_axis):])  # 更新X轴详细数据
            y_min, y_max = accresult[1, -len(detail_x_axis):].min(), accresult[1, -len(detail_x_axis):].max()
            axs[i].set_ylim(y_min, y_max)

        else:
            line.set_ydata(accresult[i, :])  # 更新X, Y, Z轴数据
            y_min, y_max = accresult[i, :].min(), accresult[i, :].max()
            axs[i].set_ylim(y_min, y_max)

    # 更新计时器文本
    if spaceDown_time is not None and timer_text is not None: # 如果按下空格键并且计时器文本不为空
        cur_time = time.time()
        information = get_TimeInterval(start_time=spaceDown_time, end_time=cur_time)
        timer_text.remove()  # 移除旧的计时器文本
        timer_text = plt.text(
            0.5, 4.75, information, horizontalalignment='center', verticalalignment='center',
            transform=plt.gca().transAxes, fontsize=20, color="m"
        )

# 窗口关闭事件处理函数
def on_close(event):
    global exit_flag
    print("绘图界面已关闭")
    exit_flag = True  # 设为True以终止主循环

# 空格键按下事件处理函数，触发定时器和数据保存功能
def down_space_save_before(event):
    global last_save_time, spaceDown_time, timer_text
    if event.key == ' ':

        time0 = time.time()
        if time0 - last_save_time > collect_time:
            # 重置计时器
            spaceDown_time = time.time()
            # 移除旧的计时器文本
            if timer_text is not None:
                timer_text.remove()
            # 初始化新的计时器文本
            timer_text = plt.text(
                0.5, 4.75, f'00:00.00', horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes, fontsize=20, color="m"
            )
            last_save_time = time.time()
            saveScgData()

# 定时保存加速度数据到文件
def saveScgData():
    global accresult, save_path, is_save_mat, timer, collect_time, count, figure_text
    now_time = get_currentTime()  # 获取当前时间
    if is_save_mat:
        # 将数据保存为.mat文件
        scipy.io.savemat(f"{save_path}/{now_time}.mat", {"accresult": accresult})
        print("保存 " + f"{save_path}/{now_time}.mat")
        count += 1  # 增加保存计数
        figure_text.remove()  # 更新保存次数的显示文本

        figure_text = plt.text(
            # 根据保存时间输出正确的时间
            0.5, 5, f'The data is saved every {collect_time}s,The number saved is {count}',
            horizontalalignment='center', verticalalignment='center',
            transform=plt.gca().transAxes, fontsize=15
        )

    # 重新启动定时器，循环保存数据
    timer = threading.Timer(collect_time, saveScgData)
    timer.start()

# 主程序入口
if __name__ == '__main__':
    fig = None
    # 忽略警告信息
    warnings.filterwarnings("ignore")

    # 用户信息
    user = "noise_phone"
    user_id = "sjx"
    save_path = "./collect_data/device3/"+user+"/"+user_id+"/scg"
    mk_ecg_path = "./collect_data/device3/"+user+"/"+user_id+"/ecg"
    os.makedirs(save_path, exist_ok=True)  # 创建保存数据的目录
    # os.makedirs(mk_ecg_path, exist_ok=True)  # 创建ECG数据目录（未使用）

    # 设置IP地址和端口号
    host = "0.0.0.0"
    port = 12345
    count = 0
    figsize = (12, 5)  # 设置图像大小
    sample_rate = 500  # 采样率设置为495Hz
    plot_stop_time = 1 / 25  # 每次画图停顿时间
    collect_time = 10  # 每次收集数据的时间为10秒
    detail_time = 5  # 详细显示时间段
    is_save_mat = True  # 是否保存为.mat文件
    time_axis = np.arange(1, sample_rate * collect_time + 1) / sample_rate  # X轴时间坐标
    detail_x_axis = np.arange(1, sample_rate * detail_time + 1) / sample_rate  # 详细显示X轴坐标
    timer = None  # 定时器
    space_pressed = False  # 空格键是否被按下
    figure_text = None  # 保存文本信息
    timer_text = None  # 计时器文本
    spaceDown_time = None  # 记录空格键按下时间

    save_avaiable = True # 是否可以保存数据
    after_save_time = 0  # 保存后的时间
    last_save_time = time.time()  # 上次保存的时间

    # 创建TCP服务器
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))  # 绑定IP和端口
    server_socket.listen(1)  # 设置最大连接数
    print("等待连接........")
    client_socket, addr = server_socket.accept()  # 等待客户端连接
    print('连接建立........')

    # 退出标志
    exit_flag = False

    # 创建图形窗口和子图
    accresult = np.zeros((4, sample_rate * collect_time))  # 初始化加速度数据矩阵
    color_arr = ["red", "red", "m", "gold"]  # 设置每条线的颜色
    dict_xyz = ["Y detail", "X", "Y", "Z"]  # 轴标签
    lines, axs, fig = initializeFigure(accresult, figsize, color_arr, dict_xyz)  # 初始化绘图

    # 初始化计时器和计时器文本
    spaceDown_time = time.time()
    timer_text = plt.text(
        0.5, 4.75, f'00:00.00', horizontalalignment='center', verticalalignment='center',
        transform=plt.gca().transAxes, fontsize=20, color="m"
    )

    # 使用交互模式进行绘图
    with plt.ion():
        while not exit_flag:  # 主循环，持续接收和绘制数据
            try:
                msg = client_socket.recv(1024 * 1024)  # 接收数据
            except ConnectionResetError:
                print("客户端断开连接")
                exit_flag = True  # 客户端断开连接，退出循环
                break

            data = process_data(msg)  # 处理接收到的数据
            if data is not None:
                accresult = update_accresult(data, accresult)  # 更新加速度数据
                update_figure(lines, axs, accresult)  # 更新图形
            else:
                print("客服端断开连接")
                exit_flag = True  # 如果没有数据，退出循环
                break
            plt.pause(plot_stop_time)  # 暂停以更新绘图

    client_socket.close()  # 关闭客户端连接
    plt.close()  # 关闭图形窗口
    if timer is not None:
        timer.cancel()  # 停止定时器
    print("程序结束")
    print(f"收集到{count}个数据")
