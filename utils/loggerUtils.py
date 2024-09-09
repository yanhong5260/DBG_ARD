# # 将输出保存到日志文件中
# from loguru import logger
# import time

# time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# logger.add(f'log/out-{time_now}.log')

# 将输出保存到日志文件中
from loguru import logger
import time


def MyLogger(name):
    time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    logger.add(f'Logs/{name}/{name}-{time_now}.log')
    return logger



# class MyLogger:

#     def __init__(self,name) -> None:
#         self.name = name
#         self.logger = None

#     def start(self):
#         time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#         print(f'log/{self.name}/out-{time_now}.log')
#         logger.add(f'log/{self.name}/out-{time_now}.log')
#         self.logger = logger

#获取文件名（含后缀）
# name=os.path.basename(__file__)
# print(name)

# #去掉文件后缀，只要文件名称
# name=os.path.basename(__file__).split(".")[0]
# print(name)


# time_now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
# logger.add(f'log/out-{time_now}.log')