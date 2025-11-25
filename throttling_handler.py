#!/usr/bin/env python3
"""
限流处理模块 - 智能检测和处理API限流错误
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
import random
import re
from dataclasses import dataclass
from enum import Enum


class ThrottlingStrategy(Enum):
    """限流处理策略"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"


@dataclass
class ThrottlingConfig:
    """限流处理配置"""
    initial_delay: float = 5.0  # 初始延迟（秒）
    max_delay: float = 300.0    # 最大延迟（秒）
    max_retries: int = 5        # 最大重试次数
    backoff_multiplier: float = 2.0  # 退避倍数
    jitter: bool = True         # 是否添加随机抖动
    strategy: ThrottlingStrategy = ThrottlingStrategy.EXPONENTIAL_BACKOFF


class ThrottlingDetector:
    """限流错误检测器"""
    
    # 常见的限流错误模式
    THROTTLING_PATTERNS = [
        r"throttlingException",
        r"Too many requests",
        r"Rate limit exceeded",
        r"Request rate too high",
        r"API rate limit",
        r"Quota exceeded",
        r"429",  # HTTP状态码
        r"TooManyRequestsException",
        r"RateLimitError",
        r"ThrottledError",
        r"wait before trying again",
        r"slow down",
        r"retry after",
    ]
    
    @classmethod
    def is_throttling_error(cls, error: Exception) -> bool:
        """
        检测是否为限流错误
        
        Args:
            error: 异常对象
            
        Returns:
            是否为限流错误
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # 检查错误消息和类型
        for pattern in cls.THROTTLING_PATTERNS:
            if re.search(pattern.lower(), error_str) or re.search(pattern.lower(), error_type):
                return True
        
        # 检查特定的异常类型
        throttling_exceptions = [
            "throttlingexception",
            "ratelimiterror", 
            "toomanyrequestsexception",
            "quotaexceedederror",
            "eventstreamError"  # AWS Bedrock特有
        ]
        
        return error_type in throttling_exceptions
    
    @classmethod
    def extract_retry_after(cls, error: Exception) -> Optional[float]:
        """
        从错误中提取建议的重试延迟时间
        
        Args:
            error: 异常对象
            
        Returns:
            建议的延迟时间（秒），如果无法提取则返回None
        """
        error_str = str(error)
        
        # 查找 "retry after X seconds" 模式
        retry_patterns = [
            r"retry after (\d+)",
            r"wait (\d+) seconds",
            r"try again in (\d+)",
            r"retry-after: (\d+)",
        ]
        
        for pattern in retry_patterns:
            match = re.search(pattern, error_str, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None


class ThrottlingHandler:
    """智能限流处理器"""
    
    def __init__(self, config: Optional[ThrottlingConfig] = None, logger: Optional[logging.Logger] = None):
        """
        初始化限流处理器
        
        Args:
            config: 限流处理配置
            logger: 日志记录器
        """
        self.config = config or ThrottlingConfig()
        self.logger = logger or logging.getLogger(__name__)
        self.retry_history: Dict[str, list] = {}  # 重试历史记录
        
    def calculate_delay(self, attempt: int, base_delay: Optional[float] = None) -> float:
        """
        计算延迟时间
        
        Args:
            attempt: 当前重试次数（从1开始）
            base_delay: 基础延迟时间
            
        Returns:
            计算出的延迟时间（秒）
        """
        base = base_delay or self.config.initial_delay
        
        if self.config.strategy == ThrottlingStrategy.EXPONENTIAL_BACKOFF:
            delay = base * (self.config.backoff_multiplier ** (attempt - 1))
        elif self.config.strategy == ThrottlingStrategy.LINEAR_BACKOFF:
            delay = base * attempt
        elif self.config.strategy == ThrottlingStrategy.FIXED_DELAY:
            delay = base
        elif self.config.strategy == ThrottlingStrategy.ADAPTIVE:
            # 自适应策略：根据历史重试情况调整
            delay = self._adaptive_delay(attempt, base)
        else:
            delay = base
        
        # 限制最大延迟
        delay = min(delay, self.config.max_delay)
        
        # 添加随机抖动
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10%的抖动
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(delay, 1.0)  # 最小延迟1秒
    
    def _adaptive_delay(self, attempt: int, base_delay: float) -> float:
        """
        自适应延迟计算
        
        Args:
            attempt: 重试次数
            base_delay: 基础延迟
            
        Returns:
            自适应延迟时间
        """
        # 基于历史重试成功率调整延迟
        if len(self.retry_history) > 0:
            avg_retries = sum(len(history) for history in self.retry_history.values()) / len(self.retry_history)
            if avg_retries > 3:
                # 如果平均重试次数较高，增加延迟
                multiplier = 1.5
            else:
                multiplier = 1.0
        else:
            multiplier = 1.0
        
        return base_delay * (self.config.backoff_multiplier ** (attempt - 1)) * multiplier
    
    async def handle_throttling_async(self, 
                                    func: Callable, 
                                    *args, 
                                    operation_id: Optional[str] = None,
                                    **kwargs) -> Any:
        """
        异步处理限流重试
        
        Args:
            func: 要执行的异步函数
            *args: 函数参数
            operation_id: 操作ID，用于跟踪重试历史
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 超过最大重试次数后抛出最后一次的异常
        """
        operation_id = operation_id or f"async_op_{int(time.time())}"
        self.retry_history[operation_id] = []
        
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"执行异步操作 {operation_id}，尝试 {attempt}/{self.config.max_retries}")
                
                # 执行函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # 成功执行，记录并返回结果
                self.retry_history[operation_id].append({
                    "attempt": attempt,
                    "success": True,
                    "timestamp": datetime.now()
                })
                
                self.logger.info(f"操作 {operation_id} 在第 {attempt} 次尝试成功")
                return result
                
            except Exception as e:
                last_exception = e
                
                # 记录失败尝试
                self.retry_history[operation_id].append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                
                # 检查是否为限流错误
                if not ThrottlingDetector.is_throttling_error(e):
                    self.logger.error(f"操作 {operation_id} 遇到非限流错误: {e}")
                    raise e
                
                # 如果是最后一次尝试，抛出异常
                if attempt >= self.config.max_retries:
                    self.logger.error(f"操作 {operation_id} 超过最大重试次数 {self.config.max_retries}")
                    break
                
                # 计算延迟时间
                suggested_delay = ThrottlingDetector.extract_retry_after(e)
                delay = self.calculate_delay(attempt, suggested_delay)
                
                self.logger.warning(
                    f"操作 {operation_id} 遇到限流错误 (尝试 {attempt}/{self.config.max_retries}): {e}"
                )
                self.logger.info(f"等待 {delay:.2f} 秒后重试...")
                
                print(f"⏳ 检测到限流错误，等待 {delay:.2f} 秒后重试... (尝试 {attempt}/{self.config.max_retries})")
                
                # 等待
                await asyncio.sleep(delay)
        
        # 所有重试都失败了
        raise last_exception
    
    def handle_throttling_sync(self, 
                              func: Callable, 
                              *args, 
                              operation_id: Optional[str] = None,
                              **kwargs) -> Any:
        """
        同步处理限流重试
        
        Args:
            func: 要执行的同步函数
            *args: 函数参数
            operation_id: 操作ID，用于跟踪重试历史
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 超过最大重试次数后抛出最后一次的异常
        """
        operation_id = operation_id or f"sync_op_{int(time.time())}"
        self.retry_history[operation_id] = []
        
        last_exception = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"执行同步操作 {operation_id}，尝试 {attempt}/{self.config.max_retries}")
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 成功执行，记录并返回结果
                self.retry_history[operation_id].append({
                    "attempt": attempt,
                    "success": True,
                    "timestamp": datetime.now()
                })
                
                self.logger.info(f"操作 {operation_id} 在第 {attempt} 次尝试成功")
                return result
                
            except Exception as e:
                last_exception = e
                
                # 记录失败尝试
                self.retry_history[operation_id].append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now()
                })
                
                # 检查是否为限流错误
                if not ThrottlingDetector.is_throttling_error(e):
                    self.logger.error(f"操作 {operation_id} 遇到非限流错误: {e}")
                    raise e
                
                # 如果是最后一次尝试，抛出异常
                if attempt >= self.config.max_retries:
                    self.logger.error(f"操作 {operation_id} 超过最大重试次数 {self.config.max_retries}")
                    break
                
                # 计算延迟时间
                suggested_delay = ThrottlingDetector.extract_retry_after(e)
                delay = self.calculate_delay(attempt, suggested_delay)
                
                self.logger.warning(
                    f"操作 {operation_id} 遇到限流错误 (尝试 {attempt}/{self.config.max_retries}): {e}"
                )
                self.logger.info(f"等待 {delay:.2f} 秒后重试...")
                
                print(f"⏳ 检测到限流错误，等待 {delay:.2f} 秒后重试... (尝试 {attempt}/{self.config.max_retries})")
                
                # 等待
                time.sleep(delay)
        
        # 所有重试都失败了
        raise last_exception
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """
        获取重试统计信息
        
        Returns:
            重试统计数据
        """
        total_operations = len(self.retry_history)
        if total_operations == 0:
            return {"total_operations": 0}
        
        total_attempts = sum(len(history) for history in self.retry_history.values())
        successful_operations = sum(1 for history in self.retry_history.values() 
                                  if history and history[-1]["success"])
        
        avg_attempts = total_attempts / total_operations
        success_rate = successful_operations / total_operations
        
        return {
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "total_attempts": total_attempts,
            "average_attempts_per_operation": avg_attempts,
            "success_rate": success_rate,
            "config": {
                "strategy": self.config.strategy.value,
                "max_retries": self.config.max_retries,
                "initial_delay": self.config.initial_delay,
                "max_delay": self.config.max_delay
            }
        }
    
    def clear_history(self):
        """清除重试历史记录"""
        self.retry_history.clear()


# 便捷函数
def with_throttling_retry(config: Optional[ThrottlingConfig] = None, 
                         logger: Optional[logging.Logger] = None):
    """
    装饰器：为函数添加限流重试功能
    
    Args:
        config: 限流处理配置
        logger: 日志记录器
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        handler = ThrottlingHandler(config, logger)
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await handler.handle_throttling_async(func, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return handler.handle_throttling_sync(func, *args, **kwargs)
            return sync_wrapper
    
    return decorator


# 全局限流处理器实例
_global_throttling_handler = None

def get_global_throttling_handler() -> ThrottlingHandler:
    """获取全局限流处理器实例"""
    global _global_throttling_handler
    if _global_throttling_handler is None:
        _global_throttling_handler = ThrottlingHandler()
    return _global_throttling_handler

def set_global_throttling_config(config: ThrottlingConfig):
    """设置全局限流处理配置"""
    global _global_throttling_handler
    _global_throttling_handler = ThrottlingHandler(config)