"""
日志配置模块
提供统一的日志配置，支持按容量轮转
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime


class LoggerConfig:
    """日志配置类"""
    
    def __init__(self, log_dir="logs", max_bytes=10*1024*1024, backup_count=5):
        """
        初始化日志配置
        
        Args:
            log_dir: 日志文件目录
            max_bytes: 单个日志文件最大字节数 (默认10MB)
            backup_count: 保留的备份文件数量 (默认5个)
        """
        self.log_dir = log_dir
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 配置日志
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        # 创建根日志器
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # 清除现有的处理器
        root_logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器 - 只显示WARNING和ERROR
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        
        # 文件处理器 - 记录所有级别的日志，按容量轮转
        log_file = os.path.join(self.log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # 添加处理器到根日志器
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # 设置第三方库的日志级别
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('cv2').setLevel(logging.WARNING)
        logging.getLogger('numpy').setLevel(logging.WARNING)
        
        # 记录日志配置信息
        logger = logging.getLogger(__name__)
        logger.info(f"日志系统已初始化 - 日志目录: {self.log_dir}")
        logger.info(f"控制台输出级别: WARNING及以上")
        logger.info(f"文件输出级别: DEBUG及以上")
        logger.info(f"日志文件轮转: {self.max_bytes}字节, 保留{self.backup_count}个备份")
    
    @staticmethod
    def get_logger(name=None):
        """
        获取日志器
        
        Args:
            name: 日志器名称，如果为None则返回调用模块的日志器
            
        Returns:
            logging.Logger: 配置好的日志器
        """
        if name is None:
            # 获取调用者的模块名
            import inspect
            frame = inspect.currentframe().f_back
            name = frame.f_globals.get('__name__', 'unknown')
        
        return logging.getLogger(name)


# 全局日志配置实例
_logger_config = None


def init_logging(log_dir="logs", max_bytes=10*1024*1024, backup_count=5):
    """
    初始化全局日志配置
    
    Args:
        log_dir: 日志文件目录
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的备份文件数量
    """
    global _logger_config
    _logger_config = LoggerConfig(log_dir, max_bytes, backup_count)


def get_logger(name=None):
    """
    获取日志器
    
    Args:
        name: 日志器名称，如果为None则返回调用模块的日志器
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    if _logger_config is None:
        # 如果未初始化，使用默认配置
        init_logging()
    
    return LoggerConfig.get_logger(name)


# 便捷函数
def log_info(message, *args, **kwargs):
    """记录INFO级别日志"""
    logger = get_logger()
    logger.info(message, *args, **kwargs)


def log_debug(message, *args, **kwargs):
    """记录DEBUG级别日志"""
    logger = get_logger()
    logger.debug(message, *args, **kwargs)


def log_warning(message, *args, **kwargs):
    """记录WARNING级别日志"""
    logger = get_logger()
    logger.warning(message, *args, **kwargs)


def log_error(message, *args, **kwargs):
    """记录ERROR级别日志"""
    logger = get_logger()
    logger.error(message, *args, **kwargs)


def log_critical(message, *args, **kwargs):
    """记录CRITICAL级别日志"""
    logger = get_logger()
    logger.critical(message, *args, **kwargs)
