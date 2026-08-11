# -*- coding: utf-8 -*-
"""
QueueMonitor: 监控队列大小的 Ray Actor
"""
import os
import ray
import time
import logging

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.1)
class QueueMonitor:
    """队列监控器，定期输出队列大小信息"""
    
    def __init__(self, queues: dict, interval: float = 5.0, log_path: str = None):
        """
        初始化队列监控器
        
        Args:
            queues: 队列字典，格式为 {name: queue}
            interval: 监控间隔（秒），默认 5.0
            log_path: 推理日志文件路径，用于写入队列信息
        """
        self.queues = queues
        self.interval = interval
        
        if log_path:
            self._configure_logger(log_path)
    
    def _configure_logger(self, log_path: str):
        """为 Ray Actor 配置文件日志（仅配置一次）"""
        if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == os.path.abspath(log_path)
               for h in logger.handlers):
            return
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handler = logging.FileHandler(log_path, encoding='utf-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    def run(self):
        """运行监控循环"""
        # 确保日志输出
        logger.info("[QueueMonitor] Started monitoring queues")
        
        while True:
            try:
                sizes = {name: q.qsize() for name, q in self.queues.items()}
                msg = f"[QueueMonitor] Queue sizes: {sizes}"
                logger.info(msg)
                time.sleep(self.interval)
            except Exception as e:
                # 不中断监控，继续下一轮
                error_msg = f"[QueueMonitor] Error: {e}"
                logger.warning(error_msg)
                time.sleep(self.interval)

