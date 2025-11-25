#!/usr/bin/env python3
"""
Swarm Logger - 多代理系统的详细执行日志模块
实现代理级别的日志记录、Swarm 执行过程的跟踪、集成 Strands 官方的调试日志、实现可配置的日志级别
"""

import os
import json
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEventType(Enum):
    """日志事件类型枚举"""
    SYSTEM_INIT = "system_init"
    AGENT_CREATED = "agent_created"
    SWARM_CREATED = "swarm_created"
    TASK_STARTED = "task_started"
    AGENT_HANDOFF = "agent_handoff"
    TOOL_EXECUTION = "tool_execution"
    AGENT_RESPONSE = "agent_response"
    TASK_COMPLETED = "task_completed"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class LogEntry:
    """日志条目数据结构"""
    timestamp: str
    level: str
    event_type: str
    agent_name: Optional[str]
    message: str
    details: Dict[str, Any]
    thread_id: str
    session_id: str


@dataclass
class AgentExecutionLog:
    """代理执行日志"""
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    input_message: str
    output_message: Optional[str]
    tools_used: List[str]
    handoff_to: Optional[str]
    handoff_reason: Optional[str]
    success: bool
    error_message: Optional[str]


@dataclass
class SwarmExecutionTrace:
    """Swarm 执行跟踪"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: Optional[float]
    user_question: str
    final_answer: Optional[str]
    agent_path: List[str]
    handoff_count: int
    total_tool_calls: int
    success: bool
    error_message: Optional[str]
    agent_logs: List[AgentExecutionLog]


class SwarmLogger:
    """多代理系统日志记录器"""
    
    def __init__(self, 
                 log_level: LogLevel = LogLevel.INFO,
                 log_to_file: bool = True,
                 log_to_console: bool = True,
                 log_directory: str = "logs",
                 enable_strands_debug: bool = False):
        """
        初始化 Swarm 日志记录器
        
        Args:
            log_level: 日志级别
            log_to_file: 是否记录到文件
            log_to_console: 是否输出到控制台
            log_directory: 日志文件目录
            enable_strands_debug: 是否启用 Strands 官方调试日志
        """
        self.log_level = log_level
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_directory = Path(log_directory)
        self.enable_strands_debug = enable_strands_debug
        
        # 创建日志目录
        self.log_directory.mkdir(exist_ok=True)
        
        # 当前会话信息
        self.session_id = self._generate_session_id()
        self.current_trace: Optional[SwarmExecutionTrace] = None
        self.agent_execution_stack: List[AgentExecutionLog] = []
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        # 设置日志记录器
        self._setup_loggers()
        
        # 性能统计
        self.performance_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_handoffs": 0,
            "total_tool_calls": 0,
            "agent_usage_count": {},
            "tool_usage_count": {},
            "average_task_duration": 0.0
        }
        
        self.log_info("SwarmLogger initialized", {
            "session_id": self.session_id,
            "log_level": self.log_level.value,
            "log_to_file": self.log_to_file,
            "log_to_console": self.log_to_console,
            "log_directory": str(self.log_directory),
            "enable_strands_debug": self.enable_strands_debug
        })
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def _setup_loggers(self):
        """设置日志记录器"""
        # 设置主日志记录器
        self.logger = logging.getLogger("swarm_logger")
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        if self.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 文件处理器
        if self.log_to_file:
            log_file = self.log_directory / f"swarm_{self.session_id}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 设置 Strands 官方调试日志
        if self.enable_strands_debug:
            strands_logger = logging.getLogger("strands")
            strands_logger.setLevel(logging.DEBUG)
            
            # 为 Strands 添加文件处理器
            if self.log_to_file:
                strands_log_file = self.log_directory / f"strands_debug_{self.session_id}.log"
                strands_handler = logging.FileHandler(strands_log_file, encoding='utf-8')
                strands_handler.setFormatter(formatter)
                strands_logger.addHandler(strands_handler)
    
    def _create_log_entry(self, 
                         level: LogLevel, 
                         event_type: LogEventType, 
                         message: str,
                         agent_name: Optional[str] = None,
                         details: Optional[Dict[str, Any]] = None) -> LogEntry:
        """创建日志条目"""
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event_type=event_type.value,
            agent_name=agent_name,
            message=message,
            details=details or {},
            thread_id=str(threading.get_ident()),
            session_id=self.session_id
        )
    
    def _log_entry(self, entry: LogEntry):
        """记录日志条目"""
        with self._lock:
            # 记录到标准日志
            log_message = f"[{entry.event_type}] {entry.message}"
            if entry.agent_name:
                log_message = f"[{entry.agent_name}] {log_message}"
            
            # 添加详细信息
            if entry.details:
                log_message += f" | Details: {json.dumps(entry.details, ensure_ascii=False)}"
            
            # 根据级别记录
            if entry.level == LogLevel.DEBUG.value:
                self.logger.debug(log_message)
            elif entry.level == LogLevel.INFO.value:
                self.logger.info(log_message)
            elif entry.level == LogLevel.WARNING.value:
                self.logger.warning(log_message)
            elif entry.level == LogLevel.ERROR.value:
                self.logger.error(log_message)
            elif entry.level == LogLevel.CRITICAL.value:
                self.logger.critical(log_message)
            
            # 保存结构化日志到 JSON 文件
            if self.log_to_file:
                self._save_structured_log(entry)
    
    def _save_structured_log(self, entry: LogEntry):
        """保存结构化日志到 JSON 文件"""
        try:
            json_log_file = self.log_directory / f"structured_{self.session_id}.jsonl"
            with open(json_log_file, 'a', encoding='utf-8') as f:
                json.dump(asdict(entry), f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save structured log: {e}")
    
    # 公共日志方法
    def log_debug(self, message: str, details: Optional[Dict[str, Any]] = None, agent_name: Optional[str] = None):
        """记录调试日志"""
        entry = self._create_log_entry(LogLevel.DEBUG, LogEventType.SYSTEM_INIT, message, agent_name, details)
        self._log_entry(entry)
    
    def log_info(self, message: str, details: Optional[Dict[str, Any]] = None, agent_name: Optional[str] = None):
        """记录信息日志"""
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.SYSTEM_INIT, message, agent_name, details)
        self._log_entry(entry)
    
    def log_warning(self, message: str, details: Optional[Dict[str, Any]] = None, agent_name: Optional[str] = None):
        """记录警告日志"""
        entry = self._create_log_entry(LogLevel.WARNING, LogEventType.ERROR_OCCURRED, message, agent_name, details)
        self._log_entry(entry)
    
    def log_error(self, message: str, details: Optional[Dict[str, Any]] = None, agent_name: Optional[str] = None):
        """记录错误日志"""
        entry = self._create_log_entry(LogLevel.ERROR, LogEventType.ERROR_OCCURRED, message, agent_name, details)
        self._log_entry(entry)
    
    def log_critical(self, message: str, details: Optional[Dict[str, Any]] = None, agent_name: Optional[str] = None):
        """记录严重错误日志"""
        entry = self._create_log_entry(LogLevel.CRITICAL, LogEventType.ERROR_OCCURRED, message, agent_name, details)
        self._log_entry(entry)
    
    # 专门的事件日志方法
    def log_system_init(self, message: str, details: Optional[Dict[str, Any]] = None):
        """记录系统初始化日志"""
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.SYSTEM_INIT, message, None, details)
        self._log_entry(entry)
    
    def log_agent_created(self, agent_name: str, details: Optional[Dict[str, Any]] = None):
        """记录代理创建日志"""
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.AGENT_CREATED, 
                                     f"Agent {agent_name} created", agent_name, details)
        self._log_entry(entry)
    
    def log_swarm_created(self, details: Optional[Dict[str, Any]] = None):
        """记录 Swarm 创建日志"""
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.SWARM_CREATED, 
                                     "Swarm instance created", None, details)
        self._log_entry(entry)
    
    def log_task_started(self, question: str, details: Optional[Dict[str, Any]] = None):
        """记录任务开始日志"""
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.TASK_STARTED, 
                                     f"Task started: {question[:100]}...", None, details)
        self._log_entry(entry)
        
        # 开始新的执行跟踪
        self.start_execution_trace(question)
    
    def log_agent_handoff(self, from_agent: str, to_agent: str, reason: str, details: Optional[Dict[str, Any]] = None):
        """记录代理移交日志"""
        message = f"Handoff from {from_agent} to {to_agent}: {reason}"
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.AGENT_HANDOFF, 
                                     message, from_agent, details)
        self._log_entry(entry)
        
        # 更新性能统计
        self.performance_stats["total_handoffs"] += 1
        
        # 更新执行跟踪
        if self.current_trace:
            self.current_trace.handoff_count += 1
            if to_agent not in self.current_trace.agent_path:
                self.current_trace.agent_path.append(to_agent)
    
    def log_tool_execution(self, agent_name: str, tool_name: str, success: bool, details: Optional[Dict[str, Any]] = None):
        """记录工具执行日志"""
        status = "succeeded" if success else "failed"
        message = f"Tool {tool_name} execution {status}"
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.TOOL_EXECUTION, 
                                     message, agent_name, details)
        self._log_entry(entry)
        
        # 更新性能统计
        self.performance_stats["total_tool_calls"] += 1
        if tool_name not in self.performance_stats["tool_usage_count"]:
            self.performance_stats["tool_usage_count"][tool_name] = 0
        self.performance_stats["tool_usage_count"][tool_name] += 1
        
        # 更新执行跟踪
        if self.current_trace:
            self.current_trace.total_tool_calls += 1
    
    def log_agent_response(self, agent_name: str, response: str, details: Optional[Dict[str, Any]] = None):
        """记录代理响应日志"""
        message = f"Agent response: {response[:200]}..."
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.AGENT_RESPONSE, 
                                     message, agent_name, details)
        self._log_entry(entry)
    
    def log_task_completed(self, success: bool, final_answer: str, details: Optional[Dict[str, Any]] = None):
        """记录任务完成日志"""
        status = "successfully" if success else "with failure"
        message = f"Task completed {status}"
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.TASK_COMPLETED, 
                                     message, None, details)
        self._log_entry(entry)
        
        # 完成执行跟踪
        self.complete_execution_trace(success, final_answer)
        
        # 更新性能统计
        self.performance_stats["total_tasks"] += 1
        if success:
            self.performance_stats["successful_tasks"] += 1
        else:
            self.performance_stats["failed_tasks"] += 1
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float], details: Optional[Dict[str, Any]] = None):
        """记录性能指标日志"""
        message = f"Performance metric {metric_name}: {value}"
        entry = self._create_log_entry(LogLevel.INFO, LogEventType.PERFORMANCE_METRIC, 
                                     message, None, details)
        self._log_entry(entry)
    
    # 执行跟踪方法
    def start_execution_trace(self, user_question: str):
        """开始执行跟踪"""
        with self._lock:
            self.current_trace = SwarmExecutionTrace(
                session_id=self.session_id,
                start_time=datetime.now(),
                end_time=None,
                total_duration=None,
                user_question=user_question,
                final_answer=None,
                agent_path=[],
                handoff_count=0,
                total_tool_calls=0,
                success=False,
                error_message=None,
                agent_logs=[]
            )
    
    def complete_execution_trace(self, success: bool, final_answer: Optional[str] = None, error_message: Optional[str] = None):
        """完成执行跟踪"""
        with self._lock:
            if self.current_trace:
                self.current_trace.end_time = datetime.now()
                self.current_trace.total_duration = (
                    self.current_trace.end_time - self.current_trace.start_time
                ).total_seconds()
                self.current_trace.success = success
                self.current_trace.final_answer = final_answer
                self.current_trace.error_message = error_message
                
                # 保存执行跟踪
                self._save_execution_trace()
                
                # 更新平均任务持续时间
                total_duration = sum([
                    trace.total_duration for trace in self._get_all_traces() 
                    if trace.total_duration is not None
                ])
                task_count = len(self._get_all_traces())
                if task_count > 0:
                    self.performance_stats["average_task_duration"] = total_duration / task_count
    
    def start_agent_execution(self, agent_name: str, input_message: str) -> str:
        """开始代理执行记录"""
        execution_id = f"{agent_name}_{datetime.now().strftime('%H%M%S_%f')}"
        
        agent_log = AgentExecutionLog(
            agent_name=agent_name,
            start_time=datetime.now(),
            end_time=None,
            duration_seconds=None,
            input_message=input_message,
            output_message=None,
            tools_used=[],
            handoff_to=None,
            handoff_reason=None,
            success=False,
            error_message=None
        )
        
        with self._lock:
            self.agent_execution_stack.append(agent_log)
            
            # 更新代理使用统计
            if agent_name not in self.performance_stats["agent_usage_count"]:
                self.performance_stats["agent_usage_count"][agent_name] = 0
            self.performance_stats["agent_usage_count"][agent_name] += 1
        
        self.log_info(f"Agent execution started", {"execution_id": execution_id}, agent_name)
        return execution_id
    
    def complete_agent_execution(self, agent_name: str, success: bool, output_message: Optional[str] = None, 
                                error_message: Optional[str] = None, handoff_to: Optional[str] = None,
                                handoff_reason: Optional[str] = None):
        """完成代理执行记录"""
        with self._lock:
            # 找到对应的代理执行记录
            agent_log = None
            for log in reversed(self.agent_execution_stack):
                if log.agent_name == agent_name and log.end_time is None:
                    agent_log = log
                    break
            
            if agent_log:
                agent_log.end_time = datetime.now()
                agent_log.duration_seconds = (agent_log.end_time - agent_log.start_time).total_seconds()
                agent_log.success = success
                agent_log.output_message = output_message
                agent_log.error_message = error_message
                agent_log.handoff_to = handoff_to
                agent_log.handoff_reason = handoff_reason
                
                # 添加到当前跟踪
                if self.current_trace:
                    self.current_trace.agent_logs.append(agent_log)
        
        status = "completed" if success else "failed"
        self.log_info(f"Agent execution {status}", {
            "duration": agent_log.duration_seconds if agent_log else None,
            "tools_used": agent_log.tools_used if agent_log else [],
            "handoff_to": handoff_to
        }, agent_name)
    
    def record_tool_usage(self, agent_name: str, tool_name: str):
        """记录工具使用"""
        with self._lock:
            # 找到当前代理的执行记录
            for log in reversed(self.agent_execution_stack):
                if log.agent_name == agent_name and log.end_time is None:
                    if tool_name not in log.tools_used:
                        log.tools_used.append(tool_name)
                    break
    
    def _save_execution_trace(self):
        """保存执行跟踪到文件"""
        if not self.current_trace or not self.log_to_file:
            return
        
        try:
            trace_file = self.log_directory / f"execution_traces_{self.session_id}.jsonl"
            with open(trace_file, 'a', encoding='utf-8') as f:
                trace_dict = asdict(self.current_trace)
                # 转换 datetime 对象为字符串
                trace_dict['start_time'] = self.current_trace.start_time.isoformat()
                if self.current_trace.end_time:
                    trace_dict['end_time'] = self.current_trace.end_time.isoformat()
                
                # 转换代理日志中的 datetime
                for agent_log in trace_dict['agent_logs']:
                    agent_log['start_time'] = agent_log['start_time'].isoformat() if isinstance(agent_log['start_time'], datetime) else agent_log['start_time']
                    if agent_log['end_time']:
                        agent_log['end_time'] = agent_log['end_time'].isoformat() if isinstance(agent_log['end_time'], datetime) else agent_log['end_time']
                
                json.dump(trace_dict, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.error(f"Failed to save execution trace: {e}")
    
    def _get_all_traces(self) -> List[SwarmExecutionTrace]:
        """获取所有执行跟踪（简化版本，实际应该从文件读取）"""
        # 这里简化实现，实际应该从 jsonl 文件读取所有跟踪记录
        return [self.current_trace] if self.current_trace else []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        return {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),  # 简化实现
            "performance_stats": self.get_performance_stats(),
            "log_files": {
                "main_log": str(self.log_directory / f"swarm_{self.session_id}.log"),
                "structured_log": str(self.log_directory / f"structured_{self.session_id}.jsonl"),
                "execution_traces": str(self.log_directory / f"execution_traces_{self.session_id}.jsonl"),
                "strands_debug": str(self.log_directory / f"strands_debug_{self.session_id}.log") if self.enable_strands_debug else None
            }
        }
    
    def set_log_level(self, level: LogLevel):
        """动态设置日志级别"""
        self.log_level = level
        self.logger.setLevel(getattr(logging, level.value))
        self.log_info(f"Log level changed to {level.value}")
    
    def cleanup(self):
        """清理日志记录器"""
        self.log_info("SwarmLogger cleanup started")
        
        # 完成当前跟踪
        if self.current_trace and not self.current_trace.end_time:
            self.complete_execution_trace(False, None, "Logger cleanup")
        
        # 保存最终性能统计
        if self.log_to_file:
            try:
                stats_file = self.log_directory / f"performance_stats_{self.session_id}.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.performance_stats, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Failed to save performance stats: {e}")
        
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
        
        self.log_info("SwarmLogger cleanup completed")


# 全局日志实例（可选）
_global_logger: Optional[SwarmLogger] = None


def get_global_logger() -> Optional[SwarmLogger]:
    """获取全局日志实例"""
    return _global_logger


def set_global_logger(logger: SwarmLogger):
    """设置全局日志实例"""
    global _global_logger
    _global_logger = logger


def create_default_logger(verbose: bool = False) -> SwarmLogger:
    """创建默认日志记录器"""
    log_level = LogLevel.DEBUG if verbose else LogLevel.INFO
    return SwarmLogger(
        log_level=log_level,
        log_to_file=True,
        log_to_console=True,
        enable_strands_debug=verbose
    )