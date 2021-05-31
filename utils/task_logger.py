import logging
import time
from utils.data_io import mkdir


class TaskLogger:
    def __init__(self, task_name, log_root=None):
        self.task_name = task_name
        self.log_root = log_root
        self.__init_root_logger()
        self.root_logger = logging.getLogger()

    def __init_child_logger(self, task_name):
        logger = logging.getLogger(task_name)
        logger.propagate = False
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            f"%(asctime)s - {task_name} - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        file_name = f"{self.log_root}{task_name}_{int(time.time())}.log"

        file_handler = logging.FileHandler(filename=file_name, mode="w")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        return logger

    def __init_root_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            f"%(asctime)s - {self.task_name} - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        if self.log_root:
            mkdir(self.log_root)

            file_name = f"{self.log_root}{self.task_name}_{int(time.time())}.log"

            file_handler = logging.FileHandler(filename=file_name, mode="w")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)

    def add_child_logger(self, child_task_name):
        task_name = f"{self.task_name}.{child_task_name}"
        self.__init_child_logger(task_name)
        return logging.getLogger(task_name)


def create_logger(task_name, log_root="../log/"):
    """
    将日志输出到日志文件和控制台
    """
    import warnings
    warnings.warn("create_logger is deprecated. Use new instead.", DeprecationWarning, stacklevel=2)

    logger = logging.getLogger(task_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f'%(asctime)s - {task_name} - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # 创建一个handler，用于写入日志文件
    mkdir(log_root)
    file_name = f"{log_root}{task_name}_{int(time.time())}.log"

    file_handler = logging.FileHandler(filename=file_name, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


if __name__ == '__main__':

    _logger = create_logger("1")
    for i in range(10):
        _logger.info('hi')
