#!/usr/bin/env python3
"""
Performance Monitor - 多代理系统性能统计和监控模块
收集每个代理的执行时间、统计代理间的 handoff 次数和路径、记录工具使用情况和 token 消耗、实现性能报告生成功能
"""

import os
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
from enum import Enum
import statistics


class MetricType(Enum):
    """指标类型枚举"""
    EXECUTION_TIME = "execution_time"
    HANDOFF_COUNT = "handoff_count"
    TOOL_USAGE = "tool_usage"
    TOKEN_CONSUMPTION = "token_consumption"
    SUCCESS_RATE = "success_rate"
    THROUGHPUT = "throughput"


@dataclass
class AgentPerformanceMetrics:
    """代理性能指标"""
    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    tools_used: Dict[str, int] = field(default_factory=dict)
    handoffs_initiated: int = 0
    handoffs_received: int = 0
    tokens_consumed: Dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    error_types: Dict[str, int] = field(default_factory=dict)
    last_execution_time: Optional[datetime] = None
    
    def update_execution_time(self, duration: float):
        """更新执行时间统计"""
        self.total_execution_time += duration
        self.min_execution_time = min(self.min_execution_time, duration)
        self.max_execution_time = max(self.max_execution_time, duration)
        if self.total_executions > 0:
            self.avg_execution_time = self.total_execution_time / self.total_executions
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    def get_total_tokens(self) -> int:
        """获取总 token 消耗"""
        return sum(self.tokens_consumed.values())


@dataclass
class HandoffPattern:
    """代理移交模式"""
    from_agent: str
    to_agent: str
    count: int = 0
    avg_duration: float = 0.0
    success_rate: float = 0.0
    reasons: Dict[str, int] = field(default_factory=dict)
    last_handoff_time: Optional[datetime] = None


@dataclass
class ToolPerformanceMetrics:
    """工具性能指标"""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time: float = 0.0
    avg_execution_time: float = 0.0
    agents_using: Set[str] = field(default_factory=set)
    error_types: Dict[str, int] = field(default_factory=dict)
    
    def get_success_rate(self) -> float:
        """获取工具成功率"""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


@dataclass
class SystemPerformanceSnapshot:
    """系统性能快照"""
    timestamp: datetime
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_task_duration: float
    active_agents: int
    total_handoffs: int
    total_tool_calls: int
    total_tokens: int
    memory_usage_mb: float
    cpu_usage_percent: float


@dataclass
class PerformanceReport:
    """性能报告"""
    report_id: str
    generation_time: datetime
    time_period: Tuple[datetime, datetime]
    summary: Dict[str, Any]
    agent_metrics: Dict[str, AgentPerformanceMetrics]
    tool_metrics: Dict[str, ToolPerformanceMetrics]
    handoff_patterns: List[HandoffPattern]
    system_snapshots: List[SystemPerformanceSnapshot]
    recommendations: List[str]


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, 
                 enable_real_time_monitoring: bool = True,
                 snapshot_interval: int = 60,  # 秒
                 max_snapshots: int = 1440,    # 24小时的分钟数
                 enable_detailed_tracking: bool = True):
        """
        初始化性能监控器
        
        Args:
            enable_real_time_monitoring: 是否启用实时监控
            snapshot_interval: 快照间隔（秒）
            max_snapshots: 最大快照数量
            enable_detailed_tracking: 是否启用详细跟踪
        """
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        self.enable_detailed_tracking = enable_detailed_tracking
        
        # 性能数据存储
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.tool_metrics: Dict[str, ToolPerformanceMetrics] = {}
        self.handoff_patterns: Dict[Tuple[str, str], HandoffPattern] = {}
        self.system_snapshots: deque = deque(maxlen=max_snapshots)
        
        # 实时跟踪数据
        self.active_executions: Dict[str, Dict[str, Any]] = {}  # execution_id -> execution_info
        self.recent_handoffs: deque = deque(maxlen=100)
        self.recent_tool_calls: deque = deque(maxlen=1000)
        
        # 统计数据
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.total_handoffs = 0
        self.total_tool_calls = 0
        self.total_tokens = 0
        self.start_time = datetime.now()
        
        # 线程安全锁
        self._lock = threading.Lock()
        
        # 启动实时监控
        if self.enable_real_time_monitoring:
            self._start_real_time_monitoring()
        
        print(f"✅ PerformanceMonitor 初始化完成")
        print(f"   实时监控: {'启用' if enable_real_time_monitoring else '禁用'}")
        print(f"   快照间隔: {snapshot_interval}秒")
        print(f"   详细跟踪: {'启用' if enable_detailed_tracking else '禁用'}")
    
    def _start_real_time_monitoring(self):
        """启动实时监控线程"""
        def monitoring_loop():
            while self.enable_real_time_monitoring:
                try:
                    self._take_system_snapshot()
                    time.sleep(self.snapshot_interval)
                except Exception as e:
                    print(f"⚠️  实时监控出错: {e}")
                    time.sleep(self.snapshot_interval)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _take_system_snapshot(self):
        """拍摄系统性能快照"""
        try:
            import psutil
            
            with self._lock:
                # 计算平均任务持续时间
                avg_duration = 0.0
                if self.total_tasks > 0:
                    total_duration = sum(
                        metrics.total_execution_time 
                        for metrics in self.agent_metrics.values()
                    )
                    avg_duration = total_duration / self.total_tasks
                
                snapshot = SystemPerformanceSnapshot(
                    timestamp=datetime.now(),
                    total_tasks=self.total_tasks,
                    successful_tasks=self.successful_tasks,
                    failed_tasks=self.failed_tasks,
                    avg_task_duration=avg_duration,
                    active_agents=len([m for m in self.agent_metrics.values() if m.total_executions > 0]),
                    total_handoffs=self.total_handoffs,
                    total_tool_calls=self.total_tool_calls,
                    total_tokens=self.total_tokens,
                    memory_usage_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                    cpu_usage_percent=psutil.Process().cpu_percent()
                )
                
                self.system_snapshots.append(snapshot)
                
        except ImportError:
            # 如果没有 psutil，使用简化版本
            with self._lock:
                avg_duration = 0.0
                if self.total_tasks > 0:
                    total_duration = sum(
                        metrics.total_execution_time 
                        for metrics in self.agent_metrics.values()
                    )
                    avg_duration = total_duration / self.total_tasks
                
                snapshot = SystemPerformanceSnapshot(
                    timestamp=datetime.now(),
                    total_tasks=self.total_tasks,
                    successful_tasks=self.successful_tasks,
                    failed_tasks=self.failed_tasks,
                    avg_task_duration=avg_duration,
                    active_agents=len([m for m in self.agent_metrics.values() if m.total_executions > 0]),
                    total_handoffs=self.total_handoffs,
                    total_tool_calls=self.total_tool_calls,
                    total_tokens=self.total_tokens,
                    memory_usage_mb=0.0,  # 无法获取
                    cpu_usage_percent=0.0  # 无法获取
                )
                
                self.system_snapshots.append(snapshot)
        except Exception as e:
            print(f"⚠️  拍摄系统快照失败: {e}")
    
    def start_task_execution(self, task_id: str, question: str) -> str:
        """开始任务执行跟踪
        
        Args:
            task_id: 任务ID
            question: 用户问题
            
        Returns:
            执行ID
        """
        execution_id = f"task_{task_id}_{int(time.time() * 1000)}"
        
        with self._lock:
            self.active_executions[execution_id] = {
                "task_id": task_id,
                "question": question,
                "start_time": datetime.now(),
                "agents_involved": [],
                "tools_used": [],
                "handoffs": []
            }
            
        return execution_id
    
    def complete_task_execution(self, execution_id: str, success: bool, 
                              final_answer: Optional[str] = None,
                              error_message: Optional[str] = None):
        """完成任务执行跟踪
        
        Args:
            execution_id: 执行ID
            success: 是否成功
            final_answer: 最终答案
            error_message: 错误消息
        """
        with self._lock:
            if execution_id in self.active_executions:
                execution_info = self.active_executions[execution_id]
                execution_info["end_time"] = datetime.now()
                execution_info["success"] = success
                execution_info["final_answer"] = final_answer
                execution_info["error_message"] = error_message
                execution_info["duration"] = (
                    execution_info["end_time"] - execution_info["start_time"]
                ).total_seconds()
                
                # 更新统计
                self.total_tasks += 1
                if success:
                    self.successful_tasks += 1
                else:
                    self.failed_tasks += 1
                
                # 移除活跃执行
                del self.active_executions[execution_id]
    
    def record_agent_execution(self, agent_name: str, duration: float, 
                             success: bool, tools_used: List[str] = None,
                             tokens_consumed: Dict[str, int] = None,
                             error_type: str = None):
        """记录代理执行性能
        
        Args:
            agent_name: 代理名称
            duration: 执行时间（秒）
            success: 是否成功
            tools_used: 使用的工具列表
            tokens_consumed: 消耗的 token 数量
            error_type: 错误类型
        """
        with self._lock:
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = AgentPerformanceMetrics(agent_name=agent_name)
            
            metrics = self.agent_metrics[agent_name]
            metrics.total_executions += 1
            metrics.last_execution_time = datetime.now()
            
            if success:
                metrics.successful_executions += 1
            else:
                metrics.failed_executions += 1
                if error_type:
                    metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            # 更新执行时间统计
            metrics.update_execution_time(duration)
            
            # 记录工具使用
            if tools_used:
                for tool in tools_used:
                    metrics.tools_used[tool] = metrics.tools_used.get(tool, 0) + 1
            
            # 记录 token 消耗
            if tokens_consumed:
                for token_type, count in tokens_consumed.items():
                    metrics.tokens_consumed[token_type] = metrics.tokens_consumed.get(token_type, 0) + count
                    self.total_tokens += count
    
    def record_handoff(self, from_agent: str, to_agent: str, reason: str = None,
                      duration: float = None, success: bool = True):
        """记录代理移交
        
        Args:
            from_agent: 源代理
            to_agent: 目标代理
            reason: 移交原因
            duration: 移交耗时
            success: 是否成功
        """
        with self._lock:
            # 更新代理指标
            if from_agent in self.agent_metrics:
                self.agent_metrics[from_agent].handoffs_initiated += 1
            
            if to_agent in self.agent_metrics:
                self.agent_metrics[to_agent].handoffs_received += 1
            
            # 更新移交模式
            pattern_key = (from_agent, to_agent)
            if pattern_key not in self.handoff_patterns:
                self.handoff_patterns[pattern_key] = HandoffPattern(
                    from_agent=from_agent,
                    to_agent=to_agent
                )
            
            pattern = self.handoff_patterns[pattern_key]
            pattern.count += 1
            pattern.last_handoff_time = datetime.now()
            
            if reason:
                pattern.reasons[reason] = pattern.reasons.get(reason, 0) + 1
            
            if duration is not None:
                # 更新平均持续时间
                total_duration = pattern.avg_duration * (pattern.count - 1) + duration
                pattern.avg_duration = total_duration / pattern.count
            
            # 更新成功率
            if success:
                pattern.success_rate = (pattern.success_rate * (pattern.count - 1) + 1.0) / pattern.count
            else:
                pattern.success_rate = (pattern.success_rate * (pattern.count - 1)) / pattern.count
            
            # 记录到最近移交
            self.recent_handoffs.append({
                "timestamp": datetime.now(),
                "from_agent": from_agent,
                "to_agent": to_agent,
                "reason": reason,
                "duration": duration,
                "success": success
            })
            
            self.total_handoffs += 1
    
    def record_tool_execution(self, tool_name: str, agent_name: str, 
                            duration: float, success: bool, error_type: str = None):
        """记录工具执行性能
        
        Args:
            tool_name: 工具名称
            agent_name: 使用工具的代理
            duration: 执行时间
            success: 是否成功
            error_type: 错误类型
        """
        with self._lock:
            if tool_name not in self.tool_metrics:
                self.tool_metrics[tool_name] = ToolPerformanceMetrics(tool_name=tool_name)
            
            metrics = self.tool_metrics[tool_name]
            metrics.total_calls += 1
            metrics.agents_using.add(agent_name)
            
            if success:
                metrics.successful_calls += 1
            else:
                metrics.failed_calls += 1
                if error_type:
                    metrics.error_types[error_type] = metrics.error_types.get(error_type, 0) + 1
            
            # 更新执行时间统计
            metrics.total_execution_time += duration
            metrics.avg_execution_time = metrics.total_execution_time / metrics.total_calls
            
            # 记录到最近工具调用
            self.recent_tool_calls.append({
                "timestamp": datetime.now(),
                "tool_name": tool_name,
                "agent_name": agent_name,
                "duration": duration,
                "success": success,
                "error_type": error_type
            })
            
            self.total_tool_calls += 1
    
    def get_agent_performance(self, agent_name: str) -> Optional[AgentPerformanceMetrics]:
        """获取代理性能指标"""
        return self.agent_metrics.get(agent_name)
    
    def get_tool_performance(self, tool_name: str) -> Optional[ToolPerformanceMetrics]:
        """获取工具性能指标"""
        return self.tool_metrics.get(tool_name)
    
    def get_handoff_patterns(self) -> List[HandoffPattern]:
        """获取移交模式列表"""
        return list(self.handoff_patterns.values())
    
    def get_top_agents_by_metric(self, metric: str, limit: int = 5) -> List[Tuple[str, float]]:
        """按指标获取排名前列的代理
        
        Args:
            metric: 指标名称 ('execution_time', 'success_rate', 'tool_usage', 'handoffs')
            limit: 返回数量限制
            
        Returns:
            (代理名, 指标值) 的列表
        """
        results = []
        
        for agent_name, metrics in self.agent_metrics.items():
            if metric == 'execution_time':
                value = metrics.avg_execution_time
            elif metric == 'success_rate':
                value = metrics.get_success_rate()
            elif metric == 'tool_usage':
                value = sum(metrics.tools_used.values())
            elif metric == 'handoffs':
                value = metrics.handoffs_initiated + metrics.handoffs_received
            else:
                continue
            
            results.append((agent_name, value))
        
        # 根据指标类型排序
        if metric == 'execution_time':
            results.sort(key=lambda x: x[1])  # 执行时间越短越好
        else:
            results.sort(key=lambda x: x[1], reverse=True)  # 其他指标越高越好
        
        return results[:limit]
    
    def get_system_health_score(self) -> float:
        """计算系统健康评分 (0-100)"""
        if not self.agent_metrics:
            return 0.0
        
        # 计算各项指标
        success_rates = [metrics.get_success_rate() for metrics in self.agent_metrics.values()]
        avg_success_rate = statistics.mean(success_rates) if success_rates else 0.0
        
        tool_success_rates = [metrics.get_success_rate() for metrics in self.tool_metrics.values()]
        avg_tool_success_rate = statistics.mean(tool_success_rates) if tool_success_rates else 1.0
        
        # 计算负载均衡度（代理使用的均匀程度）
        execution_counts = [metrics.total_executions for metrics in self.agent_metrics.values()]
        if len(execution_counts) > 1:
            load_balance = 1.0 - (statistics.stdev(execution_counts) / statistics.mean(execution_counts))
            load_balance = max(0.0, min(1.0, load_balance))
        else:
            load_balance = 1.0
        
        # 综合评分
        health_score = (
            avg_success_rate * 0.4 +           # 成功率权重 40%
            avg_tool_success_rate * 0.3 +      # 工具成功率权重 30%
            load_balance * 0.3                 # 负载均衡权重 30%
        ) * 100
        
        return min(100.0, max(0.0, health_score))
    
    def generate_performance_report(self, 
                                  time_period: Optional[Tuple[datetime, datetime]] = None,
                                  include_recommendations: bool = True) -> PerformanceReport:
        """生成性能报告
        
        Args:
            time_period: 时间范围，None表示全部时间
            include_recommendations: 是否包含优化建议
            
        Returns:
            性能报告对象
        """
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if time_period is None:
            time_period = (self.start_time, datetime.now())
        
        # 生成摘要
        summary = {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "success_rate": self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
            "total_handoffs": self.total_handoffs,
            "total_tool_calls": self.total_tool_calls,
            "total_tokens": self.total_tokens,
            "active_agents": len(self.agent_metrics),
            "active_tools": len(self.tool_metrics),
            "system_health_score": self.get_system_health_score(),
            "monitoring_duration_hours": (time_period[1] - time_period[0]).total_seconds() / 3600
        }
        
        # 生成优化建议
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations()
        
        # 过滤时间范围内的快照
        filtered_snapshots = [
            snapshot for snapshot in self.system_snapshots
            if time_period[0] <= snapshot.timestamp <= time_period[1]
        ]
        
        return PerformanceReport(
            report_id=report_id,
            generation_time=datetime.now(),
            time_period=time_period,
            summary=summary,
            agent_metrics=self.agent_metrics.copy(),
            tool_metrics=self.tool_metrics.copy(),
            handoff_patterns=list(self.handoff_patterns.values()),
            system_snapshots=filtered_snapshots,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if not self.agent_metrics:
            return ["系统尚未收集到足够的性能数据"]
        
        # 分析代理性能
        success_rates = [(name, metrics.get_success_rate()) 
                        for name, metrics in self.agent_metrics.items()]
        success_rates.sort(key=lambda x: x[1])
        
        # 低成功率代理建议
        low_success_agents = [name for name, rate in success_rates if rate < 0.8]
        if low_success_agents:
            recommendations.append(
                f"以下代理成功率较低，建议检查配置和工具: {', '.join(low_success_agents)}"
            )
        
        # 执行时间分析
        execution_times = [(name, metrics.avg_execution_time) 
                          for name, metrics in self.agent_metrics.items()]
        execution_times.sort(key=lambda x: x[1], reverse=True)
        
        if execution_times and execution_times[0][1] > 30.0:  # 超过30秒
            recommendations.append(
                f"代理 {execution_times[0][0]} 平均执行时间较长 ({execution_times[0][1]:.1f}秒)，建议优化"
            )
        
        # 工具性能分析
        tool_success_rates = [(name, metrics.get_success_rate()) 
                             for name, metrics in self.tool_metrics.items()]
        low_success_tools = [name for name, rate in tool_success_rates if rate < 0.9]
        if low_success_tools:
            recommendations.append(
                f"以下工具成功率较低，建议检查: {', '.join(low_success_tools)}"
            )
        
        # 负载均衡分析
        execution_counts = [metrics.total_executions for metrics in self.agent_metrics.values()]
        if len(execution_counts) > 1:
            max_count = max(execution_counts)
            min_count = min(execution_counts)
            if max_count > min_count * 3:  # 负载不均衡
                recommendations.append("代理负载不均衡，建议调整任务分配策略")
        
        # 移交模式分析
        if self.handoff_patterns:
            handoff_counts = [pattern.count for pattern in self.handoff_patterns.values()]
            avg_handoffs = statistics.mean(handoff_counts)
            if avg_handoffs > 10:  # 移交过于频繁
                recommendations.append("代理间移交过于频繁，建议优化任务分解策略")
        
        # 系统健康评分建议
        health_score = self.get_system_health_score()
        if health_score < 70:
            recommendations.append(f"系统健康评分较低 ({health_score:.1f}/100)，建议全面检查系统配置")
        elif health_score < 85:
            recommendations.append(f"系统健康评分中等 ({health_score:.1f}/100)，有优化空间")
        
        if not recommendations:
            recommendations.append("系统运行良好，暂无优化建议")
        
        return recommendations
    
    def export_metrics(self, output_file: str, format: str = 'json'):
        """导出性能指标
        
        Args:
            output_file: 输出文件路径
            format: 导出格式 ('json', 'csv')
        """
        try:
            if format.lower() == 'json':
                self._export_json(output_file)
            elif format.lower() == 'csv':
                self._export_csv(output_file)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            print(f"✅ 性能指标已导出到: {output_file}")
            
        except Exception as e:
            print(f"❌ 导出性能指标失败: {e}")
            raise
    
    def _export_json(self, output_file: str):
        """导出为 JSON 格式"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.start_time.isoformat(),
                "end": datetime.now().isoformat()
            },
            "summary": {
                "total_tasks": self.total_tasks,
                "successful_tasks": self.successful_tasks,
                "failed_tasks": self.failed_tasks,
                "total_handoffs": self.total_handoffs,
                "total_tool_calls": self.total_tool_calls,
                "total_tokens": self.total_tokens,
                "system_health_score": self.get_system_health_score()
            },
            "agent_metrics": {
                name: asdict(metrics) for name, metrics in self.agent_metrics.items()
            },
            "tool_metrics": {
                name: asdict(metrics) for name, metrics in self.tool_metrics.items()
            },
            "handoff_patterns": [
                asdict(pattern) for pattern in self.handoff_patterns.values()
            ]
        }
        
        # 处理 datetime 和 set 对象
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=json_serializer)
    
    def _export_csv(self, output_file: str):
        """导出为 CSV 格式（简化版本）"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 代理性能数据
            writer.writerow(["Agent Performance Metrics"])
            writer.writerow([
                "Agent Name", "Total Executions", "Success Rate", 
                "Avg Execution Time", "Tools Used", "Handoffs"
            ])
            
            for name, metrics in self.agent_metrics.items():
                writer.writerow([
                    name,
                    metrics.total_executions,
                    f"{metrics.get_success_rate():.2%}",
                    f"{metrics.avg_execution_time:.2f}s",
                    len(metrics.tools_used),
                    metrics.handoffs_initiated + metrics.handoffs_received
                ])
            
            writer.writerow([])  # 空行
            
            # 工具性能数据
            writer.writerow(["Tool Performance Metrics"])
            writer.writerow([
                "Tool Name", "Total Calls", "Success Rate", 
                "Avg Execution Time", "Agents Using"
            ])
            
            for name, metrics in self.tool_metrics.items():
                writer.writerow([
                    name,
                    metrics.total_calls,
                    f"{metrics.get_success_rate():.2%}",
                    f"{metrics.avg_execution_time:.2f}s",
                    len(metrics.agents_using)
                ])
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """获取实时统计数据"""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "active_executions": len(self.active_executions),
                "total_tasks": self.total_tasks,
                "success_rate": self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
                "total_handoffs": self.total_handoffs,
                "total_tool_calls": self.total_tool_calls,
                "system_health_score": self.get_system_health_score(),
                "active_agents": len([m for m in self.agent_metrics.values() if m.total_executions > 0]),
                "recent_handoffs": len(self.recent_handoffs),
                "recent_tool_calls": len(self.recent_tool_calls)
            }
    
    def cleanup(self):
        """清理监控器"""
        print("🧹 开始清理 PerformanceMonitor...")
        
        # 停止实时监控
        self.enable_real_time_monitoring = False
        
        # 等待监控线程结束
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # 生成最终报告
        try:
            final_report = self.generate_performance_report()
            report_file = f"final_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w', encoding='utf-8') as f:
                # 简化的序列化
                report_dict = asdict(final_report)
                
                def json_serializer(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, set):
                        return list(obj)
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                json.dump(report_dict, f, indent=2, ensure_ascii=False, default=json_serializer)
            
            print(f"✅ 最终性能报告已保存: {report_file}")
            
        except Exception as e:
            print(f"⚠️  保存最终报告失败: {e}")
        
        print("✅ PerformanceMonitor 清理完成")


# 全局性能监控实例
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> Optional[PerformanceMonitor]:
    """获取全局性能监控实例"""
    return _global_monitor


def set_global_monitor(monitor: PerformanceMonitor):
    """设置全局性能监控实例"""
    global _global_monitor
    _global_monitor = monitor


def create_default_monitor(enable_real_time: bool = True) -> PerformanceMonitor:
    """创建默认性能监控器"""
    return PerformanceMonitor(
        enable_real_time_monitoring=enable_real_time,
        snapshot_interval=60,
        enable_detailed_tracking=True
    )