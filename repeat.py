import subprocess

count = 0  # 初始化计数器
while True:
    count += 1  # 每次循环开始时计数器加1
    # 运行Python脚本
    subprocess.run(["D:/Python_Environment/Anaconda3/envs/EODP-2/python.exe", "f:/Destop/新建文件夹/%main.py"])
    # 打印已执行的轮数
    print(f"已执行 {count} 轮")