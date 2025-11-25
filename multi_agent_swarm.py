#!/usr/bin/env python3
"""
Multi-Agent Swarm System - 基于 Strands Swarm 的多代理协作系统
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from strands import Agent, tool
from strands.multiagent import Swarm
from strands_tools import (
    calculator, current_time, image_reader
)
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient
from tools.code_interpreter import AgentCoreCodeInterpreter
from tools.browser import AgentCoreBrowser
from dotenv import load_dotenv

# 导入日志模块
from swarm_logger import SwarmLogger, LogLevel, create_default_logger, set_global_logger

# 导入性能监控模块
from performance_monitor import PerformanceMonitor, create_default_monitor, set_global_monitor

# 导入限流处理模块
from throttling_handler import ThrottlingHandler, ThrottlingConfig, ThrottlingDetector, ThrottlingStrategy


load_dotenv(dotenv_path=".env")

# 配置基础日志（保持向后兼容）
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)

USE_BEDROCK = os.getenv("USE_BEDROCK") == "True"
SF_API_KEY = os.getenv("SF_API_KEY")
AWS_REGION = os.getenv("AWS_REGION")


@dataclass
class AgentSpec:
    """代理配置规格"""
    name: str
    system_prompt: str
    tools: List[Any]
    role_description: str


@dataclass
class SystemResponse:
    """系统响应模型"""
    success: bool
    answer: str
    swarm_result: Any
    duration: float
    agent_path: List[str]
    timestamp: datetime


class MultiAgentSwarm:
    """多代理协作系统 - 主要入口点"""
    
    def __init__(self, verbose: bool = False, use_bedrock: bool = USE_BEDROCK, config_file: str = "swarm_config.json"):
        """
        初始化多代理系统
        
        Args:
            verbose: 是否显示详细执行过程
            use_bedrock: 是否使用 Bedrock 模型
            config_file: Swarm 配置文件路径
        """
        self.verbose = verbose
        self.use_bedrock = use_bedrock
        self.config_file = config_file
        
        # 初始化日志系统 - 实现详细执行日志（任务5.1）
        self.logger = create_default_logger(verbose=verbose)
        set_global_logger(self.logger)
        
        # 初始化性能监控系统 - 实现性能统计和监控（任务5.2）
        self.performance_monitor = create_default_monitor(enable_real_time=verbose)
        set_global_monitor(self.performance_monitor)
        
        # 初始化限流处理系统 - 智能处理API限流错误
        throttling_config = ThrottlingConfig(
            initial_delay=5.0,
            max_delay=300.0,
            max_retries=5,
            backoff_multiplier=2.0,
            jitter=True,
            strategy=ThrottlingStrategy.EXPONENTIAL_BACKOFF
        )
        self.throttling_handler = ThrottlingHandler(throttling_config, self.logger.logger)
        
        self.logger.log_system_init("MultiAgentSwarm 初始化开始", {
            "verbose": verbose,
            "use_bedrock": use_bedrock,
            "config_file": config_file,
            "performance_monitoring": True
        })
        
        # 加载配置
        self.swarm_config = self._load_swarm_config()
        
        # 初始化模型
        self.model = self._initialize_model()
        
        # 初始化工具和MCP客户端
        self.mcp_clients = []
        self.mcp_tools = []
        self.basic_tools = []
        
        # 初始化基础工具
        self._setup_basic_tools()
        
        # 设置MCP连接
        self._setup_mcp()
        
        # 创建专业化代理
        self.agents = []
        self.swarm = None
        
        # 自动创建代理和Swarm（满足需求1.1 - 系统启动时创建代理）
        self._initialize_system()
        
        # 记录初始化完成日志
        init_summary = {
            "model": self.model.config['model_id'],
            "basic_tools_count": len(self.basic_tools),
            "mcp_tools_count": len(self.mcp_tools),
            "agents_count": len(self.agents),
            "swarm_config": self.config_file,
            "throttling_handler": {
                "enabled": hasattr(self, 'throttling_handler'),
                "initial_delay": self.throttling_handler.config.initial_delay if hasattr(self, 'throttling_handler') else None,
                "max_retries": self.throttling_handler.config.max_retries if hasattr(self, 'throttling_handler') else None,
                "strategy": self.throttling_handler.config.strategy.value if hasattr(self, 'throttling_handler') else None
            }
        }
        
        self.logger.log_system_init("MultiAgentSwarm 初始化完成", init_summary)
        
    
    def _initialize_model(self):
        """初始化模型"""
        if self.use_bedrock:
            return BedrockModel(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
                region_name=AWS_REGION, 
                temperature=0.7,          
                max_tokens=15000,
            )
        else:
            return OpenAIModel(
                client_args={
                    "api_key": SF_API_KEY,
                    "base_url": "https://api.siliconflow.cn/v1"
                },
                model_id="zai-org/GLM-4.5V",
                params={"max_tokens": 4096, "temperature": 0.7}
            )
    
    def _initialize_system(self):
        """初始化整个多代理系统 - 添加完整的错误处理和回退机制"""
        try:
            self.logger.log_system_init("开始初始化多代理系统")
            
            # 第一步：创建专业化代理
            self.logger.log_system_init("第1步：创建专业化代理")
            self.create_specialized_agents()
            
            # 第二步：创建Swarm实例
            self.logger.log_system_init("第2步：创建 Swarm 实例")
            self.create_swarm()
            
            # 第三步：验证系统状态
            self.logger.log_system_init("第3步：验证系统状态")
            self._validate_system_state()
            
            success_details = {
                "agents_count": len(self.agents),
                "swarm_created": self.swarm is not None,
                "config_file": self.config_file
            }
            
            self.logger.log_system_init("多代理系统初始化成功", success_details)
            
        except Exception as e:
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "agents_created": len(self.agents) if hasattr(self, 'agents') else 0,
                "swarm_created": hasattr(self, 'swarm') and self.swarm is not None
            }
            
            self.logger.log_error("系统初始化失败", error_details)
            
            # 清理部分创建的资源
            try:
                if hasattr(self, 'swarm') and self.swarm:
                    self.swarm = None
                if hasattr(self, 'agents') and self.agents:
                    self.agents.clear()
                self.logger.log_info("资源清理完成")
            except Exception as cleanup_error:
                self.logger.log_error("资源清理时出错", {"cleanup_error": str(cleanup_error)})
            
            # 重新抛出原始异常
            raise Exception(f"多代理系统初始化失败: {e}")
    
    def _validate_system_state(self) -> None:
        """验证系统状态是否正常"""
        errors = []
        
        # 验证代理
        if not self.agents:
            errors.append("代理列表为空")
        elif len(self.agents) < 4:
            errors.append(f"代理数量不足：需要4个，实际{len(self.agents)}个")
        
        # 验证 Swarm
        if not self.swarm:
            errors.append("Swarm 实例未创建")
        
        # 验证配置
        if not self.swarm_config:
            errors.append("Swarm 配置为空")
        
        # 验证模型
        if not self.model:
            errors.append("模型未初始化")
        
        if errors:
            raise Exception(f"系统状态验证失败: {'; '.join(errors)}")
        
    
    def _load_swarm_config(self) -> Dict[str, Any]:
        """加载 Swarm 配置 - 配置管理功能，添加错误处理和回退机制"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 验证配置完整性
                self._validate_config(config)
                
                return config
            else:
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
                
        except json.JSONDecodeError as e:
            return self._handle_config_error("json_decode_error")
            
        except ValueError as e:
            return self._handle_config_error("validation_error")
            
        except FileNotFoundError:
            default_config = self._get_default_config()
            self._save_config(default_config)
            return default_config
            
        except PermissionError:
            return self._get_default_config()
            
        except Exception as e:
            return self._get_default_config()
    
    def _handle_config_error(self, error_type: str) -> Dict[str, Any]:
        """
        处理配置错误 - 提供清晰的错误提示和自动修复建议（需求5.4）
        
        Args:
            error_type: 错误类型
            
        Returns:
            修复后的配置或默认配置
        """
        
        # 尝试备份损坏的配置文件
        if os.path.exists(self.config_file):
            backup_file = f"{self.config_file}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                import shutil
                shutil.copy2(self.config_file, backup_file)
            except Exception as e:
        
        # 根据错误类型提供修复建议
        if error_type == "json_decode_error":
            
        elif error_type == "validation_error":
        
        # 创建并保存默认配置
        default_config = self._get_default_config()
        
        try:
            self._save_config(default_config)
        except Exception as e:
        
        return default_config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置文件的完整性 - 添加配置验证和错误处理"""
        validation_errors = []
        
        # 验证顶级结构
        if "swarm_config" not in config:
            validation_errors.append("缺少 'swarm_config' 配置节")
        
        if "agents" not in config:
            validation_errors.append("缺少 'agents' 配置节")
        
        # 验证 swarm_config 必需参数
        required_swarm_keys = [
            "max_handoffs", "max_iterations", "execution_timeout", 
            "node_timeout", "repetitive_handoff_detection_window", 
            "repetitive_handoff_min_unique_agents"
        ]
        
        swarm_config = config.get("swarm_config", {})
        
        for key in required_swarm_keys:
            if key not in swarm_config:
                validation_errors.append(f"swarm_config 缺少必需项: {key}")
        
        # 验证参数类型和范围
        if "max_handoffs" in swarm_config:
            if not isinstance(swarm_config["max_handoffs"], int) or swarm_config["max_handoffs"] <= 0:
                validation_errors.append("max_handoffs 必须是正整数")
        
        if "max_iterations" in swarm_config:
            if not isinstance(swarm_config["max_iterations"], int) or swarm_config["max_iterations"] <= 0:
                validation_errors.append("max_iterations 必须是正整数")
        
        if "execution_timeout" in swarm_config:
            if not isinstance(swarm_config["execution_timeout"], (int, float)) or swarm_config["execution_timeout"] <= 0:
                validation_errors.append("execution_timeout 必须是正数")
        
        if "node_timeout" in swarm_config:
            if not isinstance(swarm_config["node_timeout"], (int, float)) or swarm_config["node_timeout"] <= 0:
                validation_errors.append("node_timeout 必须是正数")
        
        # 验证代理配置
        agents_config = config.get("agents", {})
        required_agents = ["task_analyzer", "info_gatherer", "tool_executor", "result_synthesizer"]
        
        for agent_name in required_agents:
            if agent_name not in agents_config:
                validation_errors.append(f"缺少必需的代理配置: {agent_name}")
            else:
                agent_config = agents_config[agent_name]
                if not isinstance(agent_config, dict):
                    validation_errors.append(f"代理 {agent_name} 的配置必须是字典类型")
                elif "role_description" not in agent_config:
                    validation_errors.append(f"代理 {agent_name} 缺少 role_description")
        
        # 输出验证结果
        if validation_errors:
            for error in validation_errors:
            
            # 如果是关键错误，抛出异常
            critical_errors = [e for e in validation_errors if "缺少必需" in e or "必须是" in e]
            if critical_errors:
                raise ValueError(f"配置验证失败: {'; '.join(critical_errors)}")
        else:
    
    def _save_config(self, config: Dict[str, Any]) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "swarm_config": {
                "max_handoffs": 20,
                "max_iterations": 20,
                "execution_timeout": 900.0,
                "node_timeout": 300.0,
                "repetitive_handoff_detection_window": 8,
                "repetitive_handoff_min_unique_agents": 3
            },
            "agents": {
                "task_analyzer": {
                    "role_description": "任务分析代理 - 分解复杂问题为可管理的步骤",
                    "tools": []
                },
                "info_gatherer": {
                    "role_description": "信息收集代理 - 收集和验证任务所需信息", 
                    "tools": ["image_reader", "mcp_tools"]
                },
                "tool_executor": {
                    "role_description": "工具执行代理 - 执行计算、代码运行等操作",
                    "tools": ["basic_tools", "mcp_tools"]
                },
                "result_synthesizer": {
                    "role_description": "结果分析代理 - 整合结果并生成最终答案",
                    "tools": []
                }
            }
        }
    
    def get_agent_info(self) -> List[Dict[str, Any]]:
        """获取代理信息 - 用于监控和调试"""
        agent_info = []
        for agent in self.agents:
            # 获取代理的工具数量，考虑不同的工具存储方式
            tools_count = 0
            if hasattr(agent, 'tools') and agent.tools:
                tools_count = len(agent.tools)
            elif hasattr(agent, '_tools') and agent._tools:
                tools_count = len(agent._tools)
            elif hasattr(agent, 'tool_registry') and agent.tool_registry:
                # 尝试不同的方式获取工具数量
                try:
                    if hasattr(agent.tool_registry, 'tools'):
                        tools_count = len(agent.tool_registry.tools)
                    elif hasattr(agent.tool_registry, '_tools'):
                        tools_count = len(agent.tool_registry._tools)
                    elif hasattr(agent.tool_registry, 'list_tools'):
                        tools_count = len(agent.tool_registry.list_tools())
                    else:
                        # 如果无法获取具体数量，至少表明有工具注册表
                        tools_count = 1
                except:
                    tools_count = 0
            
            info = {
                "name": agent.name if hasattr(agent, 'name') else "unknown",
                "tools_count": tools_count,
                "model": self.model.config['model_id'],
                "status": "active"
            }
            agent_info.append(info)
        return agent_info
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """获取 Swarm 状态信息"""
        return {
            "initialized": self.swarm is not None,
            "agents_count": len(self.agents),
            "config_file": self.config_file,
            "verbose_mode": self.verbose,
            "model_provider": "bedrock" if self.use_bedrock else "openai",
            "basic_tools_count": len(self.basic_tools),
            "mcp_tools_count": len(self.mcp_tools),
            "swarm_config": self.swarm_config.get("swarm_config", {}),
            "config_valid": True  # 如果能到这里说明配置是有效的
        }
    
    def update_swarm_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新 Swarm 配置 - 支持动态调整代理组合（需求5.2）
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            更新是否成功
        """
        try:
            # 验证新配置
            self._validate_config(new_config)
            
            # 备份当前配置
            old_config = self.swarm_config.copy()
            
            # 更新配置
            self.swarm_config = new_config
            
            # 保存到文件
            self._save_config(new_config)
            
            # 如果 Swarm 参数发生变化，需要重新创建 Swarm
            old_swarm_config = old_config.get("swarm_config", {})
            new_swarm_config = new_config.get("swarm_config", {})
            
            if old_swarm_config != new_swarm_config:
                self.create_swarm()
            
            return True
            
        except Exception as e:
            # 恢复原配置
            if 'old_config' in locals():
                self.swarm_config = old_config
            return False
    
    def reload_config(self) -> bool:
        """
        重新加载配置文件 - 支持配置热更新
        
        Returns:
            重新加载是否成功
        """
        try:
            new_config = self._load_swarm_config()
            
            # 检查配置是否有变化
            if new_config != self.swarm_config:
                return self.update_swarm_config(new_config)
            else:
                return True
                
        except Exception as e:
            return False
    
    def _setup_basic_tools(self):
        """设置基础工具 - 处理工具初始化失败，提供清晰的错误提示信息"""
        self.basic_tools = []
        tool_init_results = {}
        
        # 基础工具列表（总是可用的）
        basic_safe_tools = [
            ("calculator", calculator),
            ("current_time", current_time),
            ("image_reader", image_reader)
        ]
        
        # 尝试初始化基础安全工具
        for tool_name, tool_func in basic_safe_tools:
            try:
                if tool_func:
                    self.basic_tools.append(tool_func)
                    tool_init_results[tool_name] = {"status": "success", "error": None}
                else:
                    tool_init_results[tool_name] = {"status": "failed", "error": "Tool function is None"}
            except Exception as e:
                tool_init_results[tool_name] = {"status": "failed", "error": str(e)}
        
        # 尝试初始化 AgentCore 工具（可能失败）
        advanced_tools = [
            ("code_interpreter", self._init_code_interpreter),
            ("browser", self._init_browser)
        ]
        
        for tool_name, init_func in advanced_tools:
            try:
                tool = init_func()
                if tool:
                    self.basic_tools.append(tool)
                    tool_init_results[tool_name] = {"status": "success", "error": None}
                else:
                    tool_init_results[tool_name] = {"status": "failed", "error": "Tool initialization returned None"}
            except Exception as e:
                tool_init_results[tool_name] = {"status": "failed", "error": str(e)}
                self._provide_tool_error_guidance(tool_name, e)
        
        # 验证至少有基础工具可用
        if len(self.basic_tools) == 0:
            error_msg = "所有基础工具初始化失败，系统无法正常工作"
            self.logger.log_error("基础工具初始化失败", tool_init_results)
            raise Exception(error_msg)
        
        # 记录工具初始化结果
        self.logger.log_info("基础工具初始化完成", {
            "total_tools": len(self.basic_tools),
            "init_results": tool_init_results
        })
        
        
        # 如果有工具初始化失败，提供修复建议
        failed_tools = [name for name, result in tool_init_results.items() if result["status"] == "failed"]
        if failed_tools:
            self._provide_tool_recovery_suggestions(failed_tools, tool_init_results)
    
    def _init_code_interpreter(self):
        """初始化代码解释器工具"""
        try:
            agentcore_code_interpreter = AgentCoreCodeInterpreter(region="us-east-1")
            return agentcore_code_interpreter.code_interpreter
        except ImportError as e:
            return None
        except Exception as e:
            return None
    
    def _init_browser(self):
        """初始化浏览器工具"""
        try:
            agentcore_browser = AgentCoreBrowser(region="us-east-1")
            return agentcore_browser.browser
        except ImportError as e:
            return None
        except Exception as e:
            return None
    
    def _provide_tool_error_guidance(self, tool_name: str, error: Exception):
        """提供工具错误的修复指导"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        
        if error_type == "ImportError":
            if "code_interpreter" in tool_name:
            elif "browser" in tool_name:
            else:
        
        elif error_type == "ConnectionError" or "connection" in error_msg.lower():
        
        elif error_type == "PermissionError" or "permission" in error_msg.lower():
        
        elif "region" in error_msg.lower():
        
        else:
    
    def _provide_tool_recovery_suggestions(self, failed_tools: List[str], tool_results: Dict[str, Dict]):
        """提供工具恢复建议"""
        
        # 按错误类型分组建议
        import_errors = []
        connection_errors = []
        permission_errors = []
        other_errors = []
        
        for tool_name in failed_tools:
            error_msg = tool_results[tool_name]["error"]
            if "ImportError" in error_msg or "import" in error_msg.lower():
                import_errors.append(tool_name)
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                connection_errors.append(tool_name)
            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                permission_errors.append(tool_name)
            else:
                other_errors.append(tool_name)
        
        if import_errors:
        
        if connection_errors:
        
        if permission_errors:
        
        if other_errors:
        
    
    def _setup_mcp(self):
        """设置MCP连接 - 处理MCP初始化失败，提供清晰的错误提示信息"""
        self.mcp_clients = []
        self.mcp_tools = []
        mcp_init_results = {}
        
        try:
            # 检查MCP配置文件
            if not os.path.exists("mcp_config.json"):
                self._create_default_mcp_config()
                return
            
            # 读取MCP配置
            try:
                with open("mcp_config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                error_msg = f"MCP配置文件JSON格式错误: {e}"
                self._handle_mcp_config_error("json_decode_error", error_msg)
                return
            except Exception as e:
                error_msg = f"读取MCP配置文件失败: {e}"
                self._handle_mcp_config_error("file_read_error", error_msg)
                return
            
            # 验证MCP配置结构
            if not self._validate_mcp_config(config):
                return
            
            # 连接所有启用的服务器
            servers = config.get("mcpServers", {})
            if not servers:
                return
            
            
            for name, server_config in servers.items():
                if server_config.get("disabled", False):
                    mcp_init_results[name] = {"status": "disabled", "error": None}
                    continue
                
                # 尝试连接单个MCP服务器
                result = self._connect_mcp_server(name, server_config)
                mcp_init_results[name] = result
            
            # 汇总结果
            successful_servers = [name for name, result in mcp_init_results.items() if result["status"] == "success"]
            failed_servers = [name for name, result in mcp_init_results.items() if result["status"] == "failed"]
            disabled_servers = [name for name, result in mcp_init_results.items() if result["status"] == "disabled"]
            
            # 记录结果
            self.logger.log_info("MCP初始化完成", {
                "total_servers": len(servers),
                "successful": len(successful_servers),
                "failed": len(failed_servers),
                "disabled": len(disabled_servers),
                "total_tools": len(self.mcp_tools),
                "results": mcp_init_results
            })
            
            # 输出汇总信息
            if successful_servers:
            
            if failed_servers:
                self._provide_mcp_recovery_suggestions(failed_servers, mcp_init_results)
            
            if disabled_servers:
            
            if not successful_servers and not disabled_servers:
                
        except Exception as e:
            error_msg = f"MCP设置过程中发生未知错误: {e}"
            self.logger.log_error("MCP设置失败", {"error": error_msg})
            self._handle_mcp_config_error("unknown_error", error_msg)
    
    def _validate_mcp_config(self, config: Dict[str, Any]) -> bool:
        """验证MCP配置的完整性"""
        validation_errors = []
        
        # 检查顶级结构
        if not isinstance(config, dict):
            validation_errors.append("配置文件必须是JSON对象")
            
        if "mcpServers" not in config:
            validation_errors.append("缺少 'mcpServers' 配置节")
        elif not isinstance(config["mcpServers"], dict):
            validation_errors.append("'mcpServers' 必须是对象类型")
        else:
            # 验证每个服务器配置
            for server_name, server_config in config["mcpServers"].items():
                if not isinstance(server_config, dict):
                    validation_errors.append(f"服务器 '{server_name}' 配置必须是对象类型")
                    continue
                
                # 检查必需字段
                required_fields = ["command", "args"]
                for field in required_fields:
                    if field not in server_config:
                        validation_errors.append(f"服务器 '{server_name}' 缺少必需字段: {field}")
                
                # 检查字段类型
                if "command" in server_config and not isinstance(server_config["command"], str):
                    validation_errors.append(f"服务器 '{server_name}' 的 'command' 必须是字符串")
                
                if "args" in server_config and not isinstance(server_config["args"], list):
                    validation_errors.append(f"服务器 '{server_name}' 的 'args' 必须是数组")
                
                if "env" in server_config and not isinstance(server_config["env"], dict):
                    validation_errors.append(f"服务器 '{server_name}' 的 'env' 必须是对象")
                
                if "disabled" in server_config and not isinstance(server_config["disabled"], bool):
                    validation_errors.append(f"服务器 '{server_name}' 的 'disabled' 必须是布尔值")
        
        # 输出验证结果
        if validation_errors:
            for error in validation_errors:
            
            self._provide_mcp_config_fix_suggestions(validation_errors)
            return False
        else:
            return True
    
    def _connect_mcp_server(self, name: str, server_config: Dict[str, Any]) -> Dict[str, Any]:
        """连接单个MCP服务器"""
        try:
            
            # 验证服务器配置
            if not server_config.get("command"):
                return {"status": "failed", "error": "缺少command配置"}
            
            if not server_config.get("args"):
                return {"status": "failed", "error": "缺少args配置"}
            
            # 创建MCP客户端
            mcp_client = MCPClient(lambda sc=server_config: stdio_client(
                StdioServerParameters(
                    command=sc["command"],
                    args=sc["args"],
                    env=sc.get("env", {})
                )
            ))
            
            # 启动客户端
            mcp_client.start()
            
            # 获取工具列表
            tools = mcp_client.list_tools_sync()
            
            # 保存成功的连接
            self.mcp_clients.append((name, mcp_client))
            self.mcp_tools.extend(tools)
            
            
            return {
                "status": "success", 
                "error": None,
                "tools_count": len(tools),
                "command": server_config["command"],
                "args": server_config["args"]
            }
            
        except FileNotFoundError as e:
            error_msg = f"命令不存在: {server_config.get('command', 'unknown')}"
            return {"status": "failed", "error": error_msg, "error_type": "command_not_found"}
            
        except PermissionError as e:
            error_msg = f"权限不足: {e}"
            return {"status": "failed", "error": error_msg, "error_type": "permission_error"}
            
        except ConnectionError as e:
            error_msg = f"连接错误: {e}"
            return {"status": "failed", "error": error_msg, "error_type": "connection_error"}
            
        except TimeoutError as e:
            error_msg = f"连接超时: {e}"
            return {"status": "failed", "error": error_msg, "error_type": "timeout_error"}
            
        except Exception as e:
            error_msg = f"未知错误: {e}"
            return {"status": "failed", "error": error_msg, "error_type": "unknown_error"}
    
    def _handle_mcp_config_error(self, error_type: str, error_msg: str):
        """处理MCP配置错误"""
        
        if error_type == "json_decode_error":
            
        elif error_type == "file_read_error":
            
        elif error_type == "unknown_error":
        
        # 尝试创建默认配置
        self._create_default_mcp_config()
    
    def _create_default_mcp_config(self):
        """创建默认MCP配置文件"""
        try:
            default_config = {
                "mcpServers": {
                    "example-server": {
                        "command": "echo",
                        "args": ["MCP server example"],
                        "env": {},
                        "disabled": True
                    }
                }
            }
            
            with open("mcp_config.json", 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            
        except Exception as e:
    
    def _provide_mcp_config_fix_suggestions(self, validation_errors: List[str]):
        """提供MCP配置修复建议"""
        
        # 按错误类型分组建议
        if any("缺少" in error for error in validation_errors):
            
        if any("类型" in error for error in validation_errors):
        
     "mcpServers": {
       "my-server": {
         "command": "uvx",
         "args": ["my-mcp-server@latest"],
         "env": {},
         "disabled": false
       }
     }
   }""")
    
    def _provide_mcp_recovery_suggestions(self, failed_servers: List[str], results: Dict[str, Dict]):
        """提供MCP恢复建议"""
        
        # 按错误类型分组
        command_errors = []
        permission_errors = []
        connection_errors = []
        timeout_errors = []
        other_errors = []
        
        for server_name in failed_servers:
            result = results[server_name]
            error_type = result.get("error_type", "unknown")
            
            if error_type == "command_not_found":
                command_errors.append(server_name)
            elif error_type == "permission_error":
                permission_errors.append(server_name)
            elif error_type == "connection_error":
                connection_errors.append(server_name)
            elif error_type == "timeout_error":
                timeout_errors.append(server_name)
            else:
                other_errors.append(server_name)
        
        if command_errors:
        
        if permission_errors:
        
        if connection_errors:
        
        if timeout_errors:
        
        if other_errors:
        
    
    def create_specialized_agents(self) -> List[Agent]:
        """
        创建专业化代理 - 满足需求1.1, 1.2, 1.3
        
        需求1.1: 创建至少4个专业化代理
        需求1.2: 每个代理具有明确的专业领域和职责范围  
        需求1.3: 配置相应的工具集和系统提示词
        
        Returns:
            专业化代理列表
        """
        # 定义4个专业化代理规格（满足需求1.1）
        agent_specs = [
            AgentSpec(
                name="task_analyzer",
                role_description="任务分析代理 - 分解复杂问题为可管理的步骤",  # 需求1.2: 明确专业领域
                system_prompt="""你是一个任务分析专家，专门负责将复杂问题分解为可管理的步骤。

## 核心职责
1. **任务理解与分析**: 深入理解用户任务的本质、复杂度和具体要求
2. **执行计划制定**: 创建详细的、分步骤的执行计划
3. **代理协调**: 识别并决定哪些专业代理应该处理任务的不同部分
4. **工作流管理**: 协调整体工作流程和代理间的移交

## 工作流程
当你收到任务时，请按以下步骤进行：

### 第一步：任务分析
- 仔细分析用户要求什么
- 识别任务类型（计算、信息收集、数据处理、多模态分析等）
- 评估任务复杂度和所需资源
- 确定成功标准和预期输出格式

### 第二步：任务分解
- 将复杂任务分解为逻辑清晰的子步骤
- 为每个子步骤确定：
  - 具体目标
  - 所需信息或数据
  - 需要使用的工具类型
  - 预期输出

### 第三步：执行计划
- 制定详细的执行计划，包括：
  - 步骤顺序和依赖关系
  - 每个步骤的负责代理
  - 关键检查点和验证方法
  - 风险评估和备选方案

### 第四步：代理分配
根据任务需求，将工作分配给合适的专业代理：

**info_gatherer (信息收集代理)**
- 适用于：文件读取、图像分析、信息搜索、数据收集
- 移交时机：需要收集外部信息或处理多媒体内容时

**tool_executor (工具执行代理)**  
- 适用于：数学计算、代码执行、浏览器操作、数据处理
- 移交时机：需要执行具体操作或使用专业工具时

**result_synthesizer (结果综合代理)**
- 适用于：整合结果、格式化输出、生成最终答案
- 移交时机：所有必要信息已收集且操作已完成时

## 移交指南
使用 handoff 功能将控制权转移给其他代理时：
1. 清楚说明移交原因和目标
2. 提供必要的上下文信息
3. 明确指出期望的输出或下一步行动
4. 确保信息传递的完整性

## 分析原则
- **系统性思考**: 从整体到局部，确保不遗漏关键环节
- **逻辑清晰**: 步骤间的逻辑关系要明确
- **效率优先**: 选择最高效的执行路径
- **质量保证**: 在每个关键节点设置验证机制

## 输出格式
你的分析应该包括：
1. **任务理解**: 简洁描述任务本质
2. **执行计划**: 详细的分步计划
3. **代理分配**: 明确的移交决策和理由
4. **预期结果**: 描述最终期望的输出

始终保持清晰的推理过程，让用户和其他代理都能理解你的分析逻辑。""",
                tools=self._get_task_analyzer_tools()  # 需求1.3: 配置基础工具集，不包含执行工具
            ),
            
            AgentSpec(
                name="info_gatherer", 
                role_description="信息收集代理 - 收集和验证任务所需信息",  # 需求1.2: 明确专业领域
                system_prompt="""你是一个信息收集专家，专门负责收集相关数据和上下文信息。

## 核心职责
1. **多源信息收集**: 从各种来源收集信息（文件、图像、网络搜索等）
2. **信息验证**: 验证信息的准确性和相关性
3. **数据组织**: 组织和结构化收集到的数据
4. **上下文提供**: 为任务提供必要的背景信息和上下文

## 专业能力
### 多模态信息处理
- **图像分析**: 使用 image_reader 工具分析图像内容
- **文档处理**: 读取和解析各种格式的文档
- **数据提取**: 从复杂数据源中提取关键信息

### 信息验证和相关性分析
- **准确性验证**: 交叉验证信息来源，确保数据准确性
- **相关性评估**: 评估信息与任务目标的相关程度
- **质量控制**: 过滤低质量或不可靠的信息
- **完整性检查**: 确保收集的信息完整覆盖任务需求

## 工作流程
### 第一步：需求分析
- 理解任务分析代理提供的信息收集需求
- 识别需要收集的信息类型和范围
- 确定最适合的信息源和收集方法

### 第二步：信息收集
根据信息类型选择合适的工具：
- **图像内容**: 使用 image_reader 分析图像、图表、截图等
- **文档资料**: 使用 MCP 工具读取文件、搜索文档
- **网络信息**: 通过搜索工具获取在线资源
- **结构化数据**: 处理表格、数据库等结构化信息

### 第三步：信息验证
- **来源可靠性**: 评估信息来源的权威性和可信度
- **内容一致性**: 检查不同来源信息的一致性
- **时效性验证**: 确认信息的时效性和当前相关性
- **完整性评估**: 识别信息缺口和需要补充的内容

### 第四步：数据组织
- **分类整理**: 按主题、重要性、来源等维度分类
- **结构化呈现**: 以清晰的格式组织信息
- **关键信息提取**: 突出最重要和最相关的信息
- **上下文关联**: 建立信息间的逻辑关系

## 移交决策
根据收集结果决定下一步行动：

**移交给 tool_executor**
- 当需要对收集的数据进行计算或处理时
- 当需要执行特定操作来获取更多信息时
- 当需要使用专业工具分析数据时

**移交给 result_synthesizer**  
- 当信息收集完成且无需进一步处理时
- 当收集的信息已足够回答用户问题时
- 当需要直接基于收集的信息生成答案时

## 质量标准
### 信息准确性
- 优先选择权威和可靠的信息源
- 对关键信息进行多源验证
- 明确标注信息的可信度和局限性

### 相关性分析
- 严格筛选与任务目标相关的信息
- 排除冗余和无关信息
- 突出对任务完成最有价值的信息

### 完整性保证
- 确保信息覆盖任务的所有关键方面
- 识别并补充缺失的重要信息
- 提供足够的上下文信息

## 输出格式
你的信息收集结果应该包括：
1. **信息摘要**: 收集到的关键信息概述
2. **详细内容**: 按类别组织的详细信息
3. **来源标注**: 每条信息的来源和可信度评估
4. **相关性分析**: 信息与任务目标的关联度
5. **后续建议**: 基于收集结果的下一步行动建议

始终保持客观、准确、全面的信息收集标准。""",
                tools=self._get_info_gathering_tools()  # 需求1.3: 信息收集专用工具集
            ),
            
            AgentSpec(
                name="tool_executor",
                role_description="工具执行代理 - 执行计算、代码运行等操作",  # 需求1.2: 明确专业领域
                system_prompt="""你是一个工具执行专家，专门负责执行计算、代码运行和各种操作。

## 核心职责
1. **计算执行**: 使用计算工具执行数学运算和数值分析
2. **代码运行**: 执行代码并分析运行结果
3. **浏览器操作**: 执行网页交互和数据抓取
4. **数据处理**: 对数据进行转换、分析和处理

## 专业能力
### 工具选择和执行逻辑
- **计算工具**: 使用 calculator 进行数学计算和表达式求值
- **代码执行器**: 使用 code_interpreter 运行 Python 代码、数据分析、图表生成
- **浏览器工具**: 使用 browser 进行网页访问、信息抓取、表单操作
- **其他专业工具**: 根据任务需求选择最合适的 MCP 工具

### 执行策略
- **系统性执行**: 按照逻辑顺序系统地执行操作
- **步骤验证**: 在每个关键步骤后验证结果的正确性
- **错误处理**: 识别和处理执行过程中的错误
- **结果分析**: 深入分析执行结果的含义和价值

## 工作流程
### 第一步：任务理解
- 理解从其他代理接收的执行需求
- 分析需要使用的工具类型和执行策略
- 确定执行的优先级和依赖关系

### 第二步：工具选择
根据任务类型选择最适合的工具：

**数学计算任务**
- 使用 calculator 进行基础数学运算
- 使用 code_interpreter 进行复杂数值计算和统计分析

**数据处理任务**
- 使用 code_interpreter 进行数据清洗、转换、分析
- 生成图表和可视化结果
- 执行机器学习和数据科学任务

**网络操作任务**
- 使用 browser 访问网页、提取信息
- 执行表单提交、页面导航等交互操作
- 进行网络数据收集和验证

**文件和系统操作**
- 使用相应的 MCP 工具进行文件操作
- 执行系统级别的任务和配置

### 第三步：系统执行
- **分步执行**: 将复杂任务分解为可管理的步骤
- **实时监控**: 监控执行过程，及时发现问题
- **结果验证**: 在每个步骤后验证输出的正确性
- **错误恢复**: 当出现错误时，尝试替代方案或修复

### 第四步：结果分析
- **输出解释**: 详细解释执行结果的含义
- **质量评估**: 评估结果的准确性和完整性
- **影响分析**: 分析结果对整体任务的影响
- **后续建议**: 基于执行结果提出下一步建议

## 执行原则
### 准确性优先
- 选择最适合的工具完成每个任务
- 在关键步骤进行结果验证
- 对异常结果进行二次确认

### 效率优化
- 优化执行顺序，减少不必要的重复操作
- 合理利用工具的并行处理能力
- 缓存中间结果，避免重复计算

### 安全考虑
- 在执行潜在风险操作前进行安全检查
- 对敏感数据进行适当的保护
- 遵循最佳实践和安全规范

## 移交决策
根据执行结果决定下一步行动：

**继续执行**
- 当任务需要多个步骤的连续执行时
- 当需要基于前一步结果进行后续操作时

**移交给 result_synthesizer**
- 当所有必要的执行操作都已完成时
- 当执行结果需要整合和格式化时
- 当可以基于执行结果生成最终答案时

**移交给 info_gatherer**
- 当执行过程中发现需要额外信息时
- 当需要验证执行结果的准确性时

## 输出格式
你的执行结果应该包括：
1. **执行摘要**: 完成的操作和主要结果概述
2. **详细结果**: 每个步骤的具体执行结果
3. **数据输出**: 计算结果、生成的文件、图表等
4. **质量评估**: 结果的准确性和可靠性分析
5. **后续建议**: 基于执行结果的下一步行动建议

始终确保执行的系统性、准确性和可验证性。""",
                tools=self._get_execution_tools()  # 需求1.3: 完整执行工具集
            ),
            
            AgentSpec(
                name="result_synthesizer",
                role_description="结果综合代理 - 整合结果并生成最终答案",  # 需求1.2: 明确专业领域
                system_prompt="""你是一个结果综合专家，专门负责整合各代理的工作成果并生成最终格式化答案。

## 核心职责
1. **结果整合**: 整合来自所有前序代理的结果和信息
2. **答案生成**: 创建全面、格式良好的最终答案
3. **格式规范**: 确保答案符合要求的格式规范
4. **质量保证**: 提供完整、准确的响应

## 专业能力
### 最终答案格式化逻辑
- **内容整合**: 将分散的信息和结果整合为连贯的答案
- **格式标准化**: 严格按照要求的格式规范输出答案
- **质量控制**: 确保答案的完整性、准确性和可读性
- **用户体验**: 优化答案的呈现方式，提升用户体验

## 工作流程
### 第一步：信息收集和整理
- **全面回顾**: 仔细回顾所有前序代理提供的信息和结果
- **信息分类**: 将收集到的信息按重要性和相关性分类
- **关键提取**: 提取对回答用户问题最关键的信息
- **逻辑梳理**: 理清信息间的逻辑关系和因果联系

### 第二步：答案构建
- **结构设计**: 设计清晰的答案结构和逻辑框架
- **内容组织**: 将信息按逻辑顺序组织成连贯的答案
- **重点突出**: 突出最重要的结论和关键信息
- **完整性检查**: 确保答案完整回应了用户的原始问题

### 第三步：格式化处理
严格按照以下格式要求处理答案：

**必须使用 `<answer></answer>` 标签**
- 所有最终答案都必须包含在 `<answer></answer>` 标签内
- 这是系统识别最终答案的关键标识

**答案格式规范**
- **数字答案**: 如果要求数字，不使用逗号分隔符，不使用单位符号（如 $ 或 %），除非特别要求
- **字符串答案**: 如果要求字符串，不使用冠词，不使用缩写，用文字表示数字，除非特别要求
- **列表答案**: 如果要求逗号分隔的列表，根据元素类型应用上述规则
- **特定格式**: 如果要求特定的数字格式、日期格式或其他格式，严格按照要求格式化

**格式示例**
- 四舍五入到千位: `93784` 变为 `<answer>94</answer>`
- 年份中的月份: `2020-04-30` 变为 `<answer>April in 2020</answer>`
- 简单答案: `<answer>apple tree</answer>`
- 列表答案: `<answer>3, 4, 5</answer>`

### 第四步：质量验证
- **完整性验证**: 确认答案完整回应了用户问题的所有方面
- **准确性检查**: 验证答案基于可靠的信息和正确的推理
- **格式合规**: 确认答案严格符合格式要求
- **可读性优化**: 确保答案清晰易懂，逻辑连贯

## 答案质量标准
### 内容质量
- **准确性**: 基于可靠信息，避免错误和误导
- **完整性**: 全面回应用户问题，不遗漏关键信息
- **相关性**: 紧密围绕用户问题，避免无关内容
- **逻辑性**: 推理清晰，结论有据可依

### 格式质量
- **标签规范**: 严格使用 `<answer></answer>` 标签
- **格式一致**: 按照指定格式要求统一处理
- **简洁明了**: 在保证完整性的前提下尽量简洁
- **用户友好**: 考虑用户的阅读体验和理解需求

## 重要原则
### 终结性原则
- **不再移交**: 作为最后一个代理，绝不将任务移交给其他代理
- **最终负责**: 对最终答案的质量承担完全责任
- **一次完成**: 确保一次性提供完整、满意的答案

### 格式严格性
- **标签必须**: `<answer></answer>` 标签是绝对必需的
- **格式精确**: 严格按照用户要求的格式规范
- **一致性**: 在整个答案中保持格式的一致性

## 输出要求
你的最终输出必须包括：
1. **答案总结**: 简要总结基于什么信息得出了什么结论
2. **推理过程**: 简要说明得出答案的逻辑推理过程
3. **最终答案**: 使用 `<answer></answer>` 标签包含的格式化最终答案

记住：你是用户获得最终答案的最后一道关口，答案的质量直接影响用户体验。""",
                tools=self._get_result_synthesizer_tools()  # 需求1.3: 基础工具用于结果处理
            )
        ]
        
        # 验证代理数量满足需求1.1
        if len(agent_specs) < 4:
            raise Exception(f"代理数量不足：需要至少4个代理，当前只有{len(agent_specs)}个")
        
        # 创建代理实例
        agents = []
        for spec in agent_specs:
            try:
                # 根据verbose设置选择回调处理器
                if self.verbose:
                    from strands.handlers.callback_handler import PrintingCallbackHandler
                    callback_handler = PrintingCallbackHandler()
                else:
                    callback_handler = None
                
                agent = Agent(
                    model=self.model,
                    tools=spec.tools,
                    system_prompt=spec.system_prompt,  # 需求1.3: 配置系统提示词
                    callback_handler=callback_handler,
                    name=spec.name
                )
                
                agents.append(agent)
                
                # 记录代理创建日志
                agent_details = {
                    "role_description": spec.role_description,
                    "tools_count": len(spec.tools),
                    "system_prompt_length": len(spec.system_prompt),
                    "has_callback_handler": callback_handler is not None
                }
                self.logger.log_agent_created(spec.name, agent_details)
                
                
            except Exception as e:
                continue
        
        # 验证创建成功的代理数量
        if len(agents) < 4:
            raise Exception(f"代理创建失败：需要至少4个代理，实际创建了{len(agents)}个")
        
        self.agents = agents
        return agents
    
    def _get_task_analyzer_tools(self) -> List[Any]:
        """获取任务分析专用工具集 - 基础工具，不包含执行工具"""
        analyzer_tools = []
        
        # 任务分析代理只需要基础的分析工具，不需要执行工具
        # 添加时间工具用于任务规划
        if current_time:
            analyzer_tools.append(current_time)
        
        # 不添加计算器、代码执行器、浏览器等执行工具
        # 这些工具由 tool_executor 代理负责
        
        return analyzer_tools
    
    def _get_info_gathering_tools(self) -> List[Any]:
        """获取信息收集专用工具集 - 配置 image_reader 工具用于多模态信息处理"""
        info_tools = []
        
        # 添加图像读取工具用于多模态信息处理（需求4.3）
        if image_reader:
            info_tools.append(image_reader)
        
        # 添加时间工具用于信息时效性验证
        if current_time:
            info_tools.append(current_time)
        
        # 添加MCP工具（主要用于信息收集和文档处理）
        if self.mcp_tools:
            info_tools.extend(self.mcp_tools)
        
        # 不添加计算器、代码执行器等执行工具
        # 这些由 tool_executor 代理负责
        
        return info_tools
    
    def _get_execution_tools(self) -> List[Any]:
        """获取执行专用工具集 - 配置完整的工具集（code_interpreter, browser 等）"""
        execution_tools = []
        
        # 添加所有基础执行工具
        execution_tools.extend(self.basic_tools)
        
        # 确保包含关键执行工具
        tool_names = []
        for tool in self.basic_tools:
            if hasattr(tool, '__name__'):
                tool_names.append(tool.__name__)
            elif hasattr(tool, 'name'):
                tool_names.append(tool.name)
        
        
        # 添加所有MCP工具用于扩展功能
        if self.mcp_tools:
            execution_tools.extend(self.mcp_tools)
        
        
        return execution_tools
    
    def _get_result_synthesizer_tools(self) -> List[Any]:
        """获取结果综合专用工具集 - 基础工具用于结果处理"""
        synthesizer_tools = []
        
        # 结果综合代理主要负责整合和格式化，不需要执行工具
        # 添加时间工具用于时间戳和时效性标注
        if current_time:
            synthesizer_tools.append(current_time)
        
        # 不添加计算器、代码执行器、浏览器等执行工具
        # 不添加图像读取器等信息收集工具
        # 专注于结果整合和格式化
        
        
        return synthesizer_tools
    
    def create_swarm(self) -> Swarm:
        """
        创建 Swarm 实例 - 使用 strands.multiagent.Swarm 创建 swarm 实例
        配置代理列表和执行参数，实现 swarm 初始化错误处理，添加 swarm 生命周期管理
        
        满足需求1.1, 1.3, 5.4
        
        Returns:
            配置好的 Swarm 实例
        """
        # 验证代理列表（需求1.1 - 至少4个代理）
        if not self.agents:
            raise Exception("代理列表为空，无法创建 Swarm。请先调用 create_specialized_agents()")
        
        if len(self.agents) < 4:
            raise Exception(f"代理数量不足：需要至少4个专业化代理，当前只有{len(self.agents)}个")
        
        try:
            # 获取 Swarm 配置参数
            swarm_params = self.swarm_config.get("swarm_config", {})
            
            
            # 验证代理状态
            self._validate_agents_for_swarm()
            
            # 使用 Strands 官方 Swarm 实现创建实例（需求1.3）
            swarm = Swarm(
                nodes=self.agents,  # 配置代理列表
                max_handoffs=swarm_params.get("max_handoffs", 20),
                max_iterations=swarm_params.get("max_iterations", 20),
                execution_timeout=swarm_params.get("execution_timeout", 900.0),
                node_timeout=swarm_params.get("node_timeout", 300.0),
                repetitive_handoff_detection_window=swarm_params.get("repetitive_handoff_detection_window", 8),
                repetitive_handoff_min_unique_agents=swarm_params.get("repetitive_handoff_min_unique_agents", 3)
            )
            
            # 验证 Swarm 创建成功
            if not swarm:
                raise Exception("Swarm 实例创建失败：返回了空对象")
            
            # 保存 Swarm 实例
            self.swarm = swarm
            
            # 记录 Swarm 创建日志
            swarm_details = {
                "agents_count": len(self.agents),
                "agent_names": [agent.name for agent in self.agents],
                "max_handoffs": swarm_params.get("max_handoffs", 20),
                "max_iterations": swarm_params.get("max_iterations", 20),
                "execution_timeout": swarm_params.get("execution_timeout", 900.0),
                "node_timeout": swarm_params.get("node_timeout", 300.0),
                "repetitive_handoff_detection_window": swarm_params.get("repetitive_handoff_detection_window", 8),
                "repetitive_handoff_min_unique_agents": swarm_params.get("repetitive_handoff_min_unique_agents", 3)
            }
            self.logger.log_swarm_created(swarm_details)
            
            # 输出详细配置信息
            
            # 记录 Swarm 创建时间用于生命周期管理
            self._swarm_created_at = datetime.now()
            
            return swarm
            
        except ImportError as e:
            error_msg = f"Strands Swarm 导入失败: {e}。请确保已正确安装 strands 库"
            raise Exception(error_msg)
            
        except TypeError as e:
            error_msg = f"Swarm 参数配置错误: {e}。请检查配置文件中的参数类型"
            self._handle_swarm_creation_error("parameter_error", str(e))
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Swarm 创建失败: {e}"
            self._handle_swarm_creation_error("unknown_error", str(e))
            raise Exception(error_msg)
    
    def _validate_agents_for_swarm(self) -> None:
        """验证代理是否适合创建 Swarm"""
        
        required_agents = ["task_analyzer", "info_gatherer", "tool_executor", "result_synthesizer"]
        agent_names = [agent.name for agent in self.agents]
        
        # 检查必需的代理是否存在
        missing_agents = [name for name in required_agents if name not in agent_names]
        if missing_agents:
            raise Exception(f"缺少必需的代理: {missing_agents}")
        
        # 检查代理是否有有效的模型
        for agent in self.agents:
            if not hasattr(agent, 'model') or not agent.model:
                raise Exception(f"代理 {agent.name} 缺少有效的模型配置")
        
    
    def _handle_swarm_creation_error(self, error_type: str, error_details: str) -> None:
        """
        处理 Swarm 创建错误 - 实现 swarm 初始化错误处理（需求5.4）
        
        Args:
            error_type: 错误类型
            error_details: 错误详情
        """
        
        if error_type == "parameter_error":
            
        elif error_type == "agent_error":
            
        elif error_type == "unknown_error":
    
    def destroy_swarm(self) -> bool:
        """
        销毁 Swarm 实例 - Swarm 生命周期管理
        
        Returns:
            销毁是否成功
        """
        try:
            if self.swarm:
                
                # 清理 Swarm 相关资源
                self.swarm = None
                self._swarm_created_at = None
                
                return True
            else:
                return True
                
        except Exception as e:
            return False
    
    def recreate_swarm(self) -> bool:
        """
        重新创建 Swarm 实例 - 用于配置更新后的重建
        
        Returns:
            重新创建是否成功
        """
        try:
            
            # 先销毁现有实例
            self.destroy_swarm()
            
            # 清理现有代理以避免工具冲突
            if self.agents:
                self.agents.clear()
            
            # 重新创建代理
            self.create_specialized_agents()
            
            # 创建新的 Swarm 实例
            self.create_swarm()
            
            return True
            
        except Exception as e:
            return False
    
    def get_swarm_lifecycle_info(self) -> Dict[str, Any]:
        """
        获取 Swarm 生命周期信息 - 用于监控和管理
        
        Returns:
            生命周期信息字典
        """
        info = {
            "exists": self.swarm is not None,
            "created_at": getattr(self, '_swarm_created_at', None),
            "uptime_seconds": None,
            "agents_count": len(self.agents) if self.agents else 0,
            "config_file": self.config_file,
            "last_config_update": None
        }
        
        # 计算运行时间
        if hasattr(self, '_swarm_created_at') and self._swarm_created_at:
            uptime = datetime.now() - self._swarm_created_at
            info["uptime_seconds"] = uptime.total_seconds()
        
        # 获取配置文件修改时间
        if os.path.exists(self.config_file):
            try:
                config_mtime = os.path.getmtime(self.config_file)
                info["last_config_update"] = datetime.fromtimestamp(config_mtime)
            except Exception:
                pass
        
        return info
    
    def process_question(self, question: str, system_prompt: str = None) -> SystemResponse:
        """
        处理用户问题 - 主要的问题处理接口
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词（可选）
            
        Returns:
            SystemResponse 对象，包含处理结果
        """
        result = self.ask(question, system_prompt)
        
        # 将字典结果转换为 SystemResponse 对象
        return SystemResponse(
            success=result.get("success", False),
            answer=result.get("answer", ""),
            swarm_result=result.get("swarm_result"),
            duration=result.get("duration", 0.0),
            agent_path=result.get("agent_path", []),
            timestamp=result.get("timestamp", datetime.now())
        )
    
    def ask(self, question: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        向多代理系统提问 - 包含完整的错误处理和回退机制
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词（暂时保留兼容性，实际使用代理专用提示词）
            
        Returns:
            包含回答和元数据的字典
        """
        import asyncio
        
        start_time = datetime.now()
        execution_id = None
        
        try:
            # 记录任务开始日志
            task_details = {
                "question_length": len(question),
                "has_system_prompt": system_prompt is not None,
                "swarm_ready": self.swarm is not None
            }
            self.logger.log_task_started(question, task_details)
            
            # 开始性能监控
            task_id = f"task_{int(start_time.timestamp() * 1000)}"
            execution_id = self.performance_monitor.start_task_execution(task_id, question)
            
            # 确保 Swarm 已创建
            if not self.swarm:
                self.logger.log_info("Swarm 未创建，正在创建...")
                self.create_swarm()
            
            # 执行多代理协作，包含错误处理和回退机制
            return self._execute_swarm_with_fallback(question, execution_id, start_time)
            
        except Exception as e:
            # 处理顶级异常，记录错误并尝试回退
            return self._handle_critical_error(e, question, execution_id, start_time)
    
    def _execute_swarm_with_fallback(self, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        执行 Swarm 查询，包含错误处理和回退机制
        
        Args:
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            执行结果字典
        """
        import asyncio
        
        # 尝试多代理协作
        try:
            self.logger.log_info("启动多代理协作", {"question_preview": question[:100]})
            
            # 定义异步执行函数
            async def execute_swarm():
                return await asyncio.wait_for(
                    self.swarm.invoke_async(question),
                    timeout=self.swarm_config.get("swarm_config", {}).get("execution_timeout", 900.0)
                )
            
            # 使用限流处理器执行，自动处理限流错误
            swarm_result = asyncio.run(
                self.throttling_handler.handle_throttling_async(
                    execute_swarm,
                    operation_id=f"swarm_main_{execution_id}"
                )
            )
            
            # 处理 Swarm 执行结果
            return self._process_swarm_result(swarm_result, question, execution_id, start_time)
            
        except asyncio.TimeoutError:
            # 处理 Swarm 执行超时
            return self._handle_swarm_timeout(question, execution_id, start_time)
            
        except Exception as swarm_error:
            # 检查是否为限流错误（虽然应该已经被处理了，但作为额外保护）
            if ThrottlingDetector.is_throttling_error(swarm_error):
                self.logger.log_warning("限流错误未被自动处理，手动处理", {"error": str(swarm_error)})
                return self._handle_throttling_error(swarm_error, question, execution_id, start_time)
            
            # 处理其他 Swarm 执行错误，尝试回退策略
            return self._handle_swarm_error(swarm_error, question, execution_id, start_time)
    
    def _process_swarm_result(self, swarm_result: Any, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        处理 Swarm 执行结果
        
        Args:
            swarm_result: Swarm 执行结果
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            处理后的结果字典
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 添加详细的调试日志
        self.logger.log_info("开始处理 Swarm 结果", {
            "execution_id": execution_id,
            "swarm_result_type": type(swarm_result).__name__,
            "has_status": hasattr(swarm_result, 'status'),
            "has_results": hasattr(swarm_result, 'results')
        })
        
        # 提取最终答案
        final_message = ""
        success = False
        
        if hasattr(swarm_result, 'status'):
            success = swarm_result.status.value == 'completed'
            self.logger.log_info("Swarm 状态检查", {
                "status": swarm_result.status.value,
                "success": success,
                "execution_id": execution_id
            })
            
            if success:
                # 从结果中提取最终消息 - 改进的提取逻辑
                final_message = self._extract_final_answer(swarm_result, execution_id)
                
                if not final_message:
                    final_message = "任务已完成，但未获取到具体结果"
                    self.logger.log_warning("未能提取到最终答案", {
                        "execution_id": execution_id,
                        "swarm_result_structure": self._debug_swarm_result_structure(swarm_result)
                    })
            else:
                # Swarm 执行失败，记录状态并尝试回退
                self.logger.log_error("Swarm 执行失败", {
                    "status": swarm_result.status.value,
                    "question": question[:100],
                    "execution_id": execution_id
                })
                
                # 尝试从部分结果中提取信息
                final_message = self._extract_partial_results(swarm_result, execution_id)
        else:
            final_message = str(swarm_result)
            success = True  # 假设成功，如果没有状态信息
            self.logger.log_info("Swarm 结果无状态信息，直接转换为字符串", {
                "result_length": len(final_message),
                "execution_id": execution_id
            })
        
        # 提取代理执行路径
        agent_path = []
        if hasattr(swarm_result, 'node_history'):
            agent_path = [node.name for node in swarm_result.node_history if hasattr(node, 'name')]
        
        # 获取使用统计
        usage = {}
        if hasattr(swarm_result, 'accumulated_usage'):
            usage = swarm_result.accumulated_usage
        
        # 完成性能监控
        self.performance_monitor.complete_task_execution(
            execution_id, success, final_message, 
            None if success else "Swarm execution failed"
        )
        
        # 记录代理执行性能
        self._record_agent_performance(agent_path, duration, success, usage)
        
        # 记录代理移交
        self._record_agent_handoffs(agent_path, success)
        
        # 记录任务完成日志
        completion_details = {
            "duration": duration,
            "agent_path": agent_path,
            "handoff_count": len(agent_path) - 1 if len(agent_path) > 1 else 0,
            "final_answer_length": len(final_message),
            "usage": usage,
            "execution_id": execution_id
        }
        self.logger.log_task_completed(success, final_message, completion_details)
        
        # 记录性能指标
        self.logger.log_performance_metric("task_duration", duration, {
            "question_length": len(question),
            "agent_count": len(agent_path),
            "success": success,
            "execution_id": execution_id
        })
        
        # 记录最终结果
        self.logger.log_info("Swarm 结果处理完成", {
            "success": success,
            "final_message_length": len(final_message),
            "agent_path": agent_path,
            "duration": duration,
            "execution_id": execution_id
        })
        
        return {
            "success": success,
            "answer": final_message,
            "swarm_result": swarm_result,
            "duration": duration,
            "agent_path": agent_path,
            "usage": usage,
            "timestamp": end_time.isoformat(),
            "execution_mode": "multi_agent"
        }
    
    def _extract_final_answer(self, swarm_result: Any, execution_id: str) -> str:
        """
        改进的答案提取逻辑 - 支持多种答案格式
        
        Args:
            swarm_result: Swarm 执行结果
            execution_id: 执行ID
            
        Returns:
            提取的最终答案
        """
        final_message = ""
        
        # 尝试多种方式提取答案
        extraction_attempts = []
        
        if hasattr(swarm_result, 'results') and swarm_result.results:
            self.logger.log_info("尝试从 results 中提取答案", {
                "results_count": len(swarm_result.results),
                "result_keys": list(swarm_result.results.keys()),
                "execution_id": execution_id
            })
            
            # 方法1: 优先从 result_synthesizer 提取
            if 'result_synthesizer' in swarm_result.results:
                synthesizer_result = swarm_result.results['result_synthesizer']
                final_message = self._extract_from_node_result(synthesizer_result, 'result_synthesizer', execution_id)
                if final_message:
                    extraction_attempts.append("result_synthesizer")
            
            # 方法2: 如果没有找到，从最后一个代理提取
            if not final_message:
                for node_name, node_result in reversed(list(swarm_result.results.items())):
                    final_message = self._extract_from_node_result(node_result, node_name, execution_id)
                    if final_message:
                        extraction_attempts.append(f"last_agent_{node_name}")
                        break
            
            # 方法3: 如果还没找到，尝试所有代理
            if not final_message:
                for node_name, node_result in swarm_result.results.items():
                    final_message = self._extract_from_node_result(node_result, node_name, execution_id)
                    if final_message:
                        extraction_attempts.append(f"any_agent_{node_name}")
                        break
        
        # 方法4: 尝试从其他可能的属性提取
        if not final_message:
            for attr_name in ['message', 'content', 'response', 'output', 'text']:
                if hasattr(swarm_result, attr_name):
                    attr_value = getattr(swarm_result, attr_name)
                    if attr_value and str(attr_value).strip():
                        final_message = str(attr_value)
                        extraction_attempts.append(f"direct_attr_{attr_name}")
                        break
        
        # 处理 <answer></answer> 标签
        if final_message:
            final_message = self._extract_answer_from_tags(final_message, execution_id)
        
        # 记录提取结果
        self.logger.log_info("答案提取完成", {
            "success": bool(final_message),
            "extraction_methods_tried": extraction_attempts,
            "final_message_length": len(final_message) if final_message else 0,
            "execution_id": execution_id
        })
        
        return final_message
    
    def _extract_from_node_result(self, node_result: Any, node_name: str, execution_id: str) -> str:
        """
        从单个节点结果中提取答案
        
        Args:
            node_result: 节点结果
            node_name: 节点名称
            execution_id: 执行ID
            
        Returns:
            提取的答案
        """
        message = ""
        
        try:
            # 尝试多种属性路径
            extraction_paths = [
                lambda nr: nr.result.message if hasattr(nr, 'result') and hasattr(nr.result, 'message') else None,
                lambda nr: nr.result.content if hasattr(nr, 'result') and hasattr(nr.result, 'content') else None,
                lambda nr: nr.result if hasattr(nr, 'result') else None,
                lambda nr: nr.message if hasattr(nr, 'message') else None,
                lambda nr: nr.content if hasattr(nr, 'content') else None,
                lambda nr: nr.response if hasattr(nr, 'response') else None,
                lambda nr: str(nr) if nr else None
            ]
            
            for i, path_func in enumerate(extraction_paths):
                try:
                    result = path_func(node_result)
                    if result and str(result).strip():
                        message = str(result)
                        self.logger.log_info(f"从 {node_name} 提取答案成功", {
                            "extraction_path": i,
                            "message_length": len(message),
                            "execution_id": execution_id
                        })
                        break
                except Exception as e:
                    continue
            
            # 如果是列表或其他复杂类型，尝试处理
            if not message and hasattr(node_result, 'result'):
                result_obj = node_result.result
                if isinstance(result_obj, list) and result_obj:
                    # 如果是列表，取最后一个元素
                    last_item = result_obj[-1]
                    if hasattr(last_item, 'message'):
                        message = str(last_item.message)
                    elif hasattr(last_item, 'content'):
                        message = str(last_item.content)
                    else:
                        message = str(last_item)
                    
                    self.logger.log_info(f"从 {node_name} 的列表结果中提取答案", {
                        "list_length": len(result_obj),
                        "message_length": len(message),
                        "execution_id": execution_id
                    })
        
        except Exception as e:
            self.logger.log_error(f"从 {node_name} 提取答案时出错", {
                "error": str(e),
                "node_result_type": type(node_result).__name__,
                "execution_id": execution_id
            })
        
        return message
    
    def _extract_answer_from_tags(self, text: str, execution_id: str) -> str:
        """
        从文本中提取 <answer></answer> 标签内的内容
        
        Args:
            text: 原始文本
            execution_id: 执行ID
            
        Returns:
            提取的答案内容
        """
        import re
        
        # 尝试提取 <answer></answer> 标签内容
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # 取最后一个匹配的答案
            extracted_answer = matches[-1].strip()
            self.logger.log_info("从标签中提取答案成功", {
                "original_length": len(text),
                "extracted_length": len(extracted_answer),
                "matches_count": len(matches),
                "execution_id": execution_id
            })
            return extracted_answer
        else:
            # 如果没有找到标签，返回原始文本
            self.logger.log_info("未找到答案标签，返回原始文本", {
                "text_length": len(text),
                "execution_id": execution_id
            })
            return text
    
    def _extract_partial_results(self, swarm_result: Any, execution_id: str) -> str:
        """
        从部分结果中提取信息
        
        Args:
            swarm_result: Swarm 执行结果
            execution_id: 执行ID
            
        Returns:
            部分结果信息
        """
        if hasattr(swarm_result, 'results') and swarm_result.results:
            partial_results = []
            for node_name, node_result in swarm_result.results.items():
                result_text = self._extract_from_node_result(node_result, node_name, execution_id)
                if result_text:
                    partial_results.append(f"{node_name}: {result_text[:200]}")
            
            if partial_results:
                return f"任务部分完成。部分结果：\n" + "\n".join(partial_results)
            else:
                return f"任务执行状态: {swarm_result.status.value}"
        else:
            return f"任务执行状态: {swarm_result.status.value}"
    
    def _debug_swarm_result_structure(self, swarm_result: Any) -> Dict[str, Any]:
        """
        调试 Swarm 结果结构
        
        Args:
            swarm_result: Swarm 执行结果
            
        Returns:
            结构信息字典
        """
        structure_info = {
            "type": type(swarm_result).__name__,
            "attributes": [],
            "results_info": {}
        }
        
        # 获取所有属性
        for attr in dir(swarm_result):
            if not attr.startswith('_'):
                try:
                    value = getattr(swarm_result, attr)
                    if not callable(value):
                        structure_info["attributes"].append({
                            "name": attr,
                            "type": type(value).__name__,
                            "has_value": bool(value)
                        })
                except:
                    continue
        
        # 详细分析 results
        if hasattr(swarm_result, 'results') and swarm_result.results:
            for node_name, node_result in swarm_result.results.items():
                node_info = {
                    "type": type(node_result).__name__,
                    "attributes": []
                }
                
                for attr in dir(node_result):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(node_result, attr)
                            if not callable(value):
                                node_info["attributes"].append({
                                    "name": attr,
                                    "type": type(value).__name__,
                                    "has_value": bool(value)
                                })
                        except:
                            continue
                
                structure_info["results_info"][node_name] = node_info
        
        return structure_info
    
    def _handle_swarm_timeout(self, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        处理 Swarm 执行超时情况 - 实现超时处理和回退策略
        
        Args:
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            回退处理结果
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        timeout_value = self.swarm_config.get("swarm_config", {}).get("execution_timeout", 900.0)
        
        error_details = {
            "error_type": "timeout",
            "timeout_seconds": timeout_value,
            "actual_duration": duration,
            "question": question[:100]
        }
        
        self.logger.log_error("Swarm 执行超时", error_details)
        
        # 完成性能监控（标记为失败）
        self.performance_monitor.complete_task_execution(
            execution_id, False, "Execution timeout", "Swarm execution timeout"
        )
        
        # 尝试单代理模式回退
        fallback_result = self._fallback_to_single_agent(question, execution_id, start_time, "timeout")
        
        # 如果回退也失败，返回超时错误信息
        if not fallback_result["success"]:
            return {
                "success": False,
                "answer": f"任务执行超时（{timeout_value}秒），单代理回退也失败。请尝试简化问题或增加超时时间。",
                "swarm_result": None,
                "duration": duration,
                "agent_path": [],
                "usage": {},
                "timestamp": end_time.isoformat(),
                "execution_mode": "timeout_failed",
                "error_type": "timeout",
                "fallback_attempted": True
            }
        
        return fallback_result
    
    def _handle_swarm_error(self, error: Exception, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        处理 Swarm 执行错误，实现代理失败时的回退策略
        
        Args:
            error: 异常对象
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            错误处理结果
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration": duration,
            "question": question[:100],
            "is_throttling": ThrottlingDetector.is_throttling_error(error)
        }
        
        self.logger.log_error("Swarm 执行错误", error_details)
        
        # 检查是否为限流错误
        if ThrottlingDetector.is_throttling_error(error):
            return self._handle_throttling_error(error, question, execution_id, start_time)
        
        # 完成性能监控（标记为失败）
        self.performance_monitor.complete_task_execution(
            execution_id, False, str(error), f"Swarm execution error: {type(error).__name__}"
        )
        
        # 根据错误类型选择不同的回退策略
        if "agent" in str(error).lower() or "handoff" in str(error).lower():
            return self._fallback_to_single_agent(question, execution_id, start_time, "agent_error")
        
        elif "timeout" in str(error).lower():
            return self._fallback_to_single_agent(question, execution_id, start_time, "timeout")
        
        elif "model" in str(error).lower() or "api" in str(error).lower():
            try:
                # 尝试重新初始化模型
                self.model = self._initialize_model()
                return self._fallback_to_single_agent(question, execution_id, start_time, "model_error")
            except Exception as reinit_error:
                self.logger.log_error("模型重新初始化失败", {"error": str(reinit_error)})
        
        # 默认回退策略
        return self._fallback_to_single_agent(question, execution_id, start_time, "general_error")
    
    def _handle_throttling_error(self, error: Exception, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        处理限流错误 - 智能重试机制
        
        Args:
            error: 限流异常对象
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            重试结果或回退结果
        """
        import asyncio
        
        self.logger.log_warning("检测到限流错误，启动智能重试", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "execution_id": execution_id
        })
        
        
        try:
            # 使用限流处理器进行智能重试
            async def retry_swarm_execution():
                return await self.swarm.invoke_async(question)
            
            # 异步重试执行
            swarm_result = asyncio.run(
                self.throttling_handler.handle_throttling_async(
                    retry_swarm_execution,
                    operation_id=f"swarm_retry_{execution_id}"
                )
            )
            
            return self._process_swarm_result(swarm_result, question, execution_id, start_time)
            
        except Exception as retry_error:
            # 重试失败，记录统计信息并回退
            retry_stats = self.throttling_handler.get_retry_statistics()
            
            self.logger.log_error("限流重试最终失败", {
                "original_error": str(error),
                "retry_error": str(retry_error),
                "retry_statistics": retry_stats,
                "execution_id": execution_id
            })
            
            
            # 完成性能监控（标记为失败）
            self.performance_monitor.complete_task_execution(
                execution_id, False, str(retry_error), f"Throttling retry failed: {type(retry_error).__name__}"
            )
            
            # 回退到单代理模式
            return self._fallback_to_single_agent(question, execution_id, start_time, "throttling_retry_failed")
    
    def _fallback_to_single_agent(self, question: str, execution_id: str, start_time: datetime, reason: str) -> Dict[str, Any]:
        """
        单代理模式的自动回退 - 实现优雅的错误恢复机制
        
        Args:
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            reason: 回退原因
            
        Returns:
            单代理执行结果
        """
        try:
            fallback_start = datetime.now()
            
            self.logger.log_info("启动单代理回退模式", {
                "reason": reason,
                "question": question[:100],
                "execution_id": execution_id
            })
            
            
            # 创建单一的综合代理用于回退
            fallback_agent = self._create_fallback_agent()
            
            # 使用单代理执行任务
            if self.verbose:
                from strands.handlers.callback_handler import PrintingCallbackHandler
                callback_handler = PrintingCallbackHandler()
            else:
                callback_handler = None
            
            # 执行单代理查询
            response = fallback_agent.invoke(question)
            
            fallback_end = datetime.now()
            fallback_duration = (fallback_end - fallback_start).total_seconds()
            total_duration = (fallback_end - start_time).total_seconds()
            
            # 提取回答
            final_answer = ""
            if hasattr(response, 'message'):
                final_answer = str(response.message)
            else:
                final_answer = str(response)
            
            # 记录回退成功
            fallback_details = {
                "reason": reason,
                "fallback_duration": fallback_duration,
                "total_duration": total_duration,
                "answer_length": len(final_answer),
                "execution_id": execution_id
            }
            
            self.logger.log_info("单代理回退成功", fallback_details)
            
            # 更新性能监控
            self.performance_monitor.complete_task_execution(
                execution_id, True, final_answer, None
            )
            
            # 记录单代理执行性能
            self.performance_monitor.record_agent_execution(
                agent_name="fallback_agent",
                duration=fallback_duration,
                success=True,
                tools_used=[],
                tokens_consumed={}
            )
            
            return {
                "success": True,
                "answer": final_answer,
                "swarm_result": None,
                "duration": total_duration,
                "agent_path": ["fallback_agent"],
                "usage": {},
                "timestamp": fallback_end.isoformat(),
                "execution_mode": "single_agent_fallback",
                "fallback_reason": reason,
                "fallback_duration": fallback_duration
            }
            
        except Exception as fallback_error:
            # 单代理回退也失败
            fallback_end = datetime.now()
            total_duration = (fallback_end - start_time).total_seconds()
            
            error_details = {
                "fallback_reason": reason,
                "fallback_error": str(fallback_error),
                "total_duration": total_duration,
                "execution_id": execution_id
            }
            
            self.logger.log_error("单代理回退失败", error_details)
            
            # 更新性能监控
            self.performance_monitor.complete_task_execution(
                execution_id, False, str(fallback_error), "Single agent fallback failed"
            )
            
            return {
                "success": False,
                "answer": f"多代理系统执行失败 (原因: {reason})，单代理回退也失败 (错误: {fallback_error})。请检查系统配置或简化问题。",
                "swarm_result": None,
                "duration": total_duration,
                "agent_path": [],
                "usage": {},
                "timestamp": fallback_end.isoformat(),
                "execution_mode": "fallback_failed",
                "fallback_reason": reason,
                "fallback_error": str(fallback_error)
            }
    
    def _create_fallback_agent(self) -> Agent:
        """
        创建用于回退的单一综合代理
        
        Returns:
            配置好的回退代理
        """
        # 综合系统提示词，包含所有专业代理的能力
        fallback_prompt = """你是一个综合AI助手，具备多种专业能力。你需要独立完成用户的任务，包括：

## 核心能力
1. **任务分析**: 理解和分解复杂问题
2. **信息收集**: 收集和验证所需信息
3. **工具执行**: 使用各种工具执行计算和操作
4. **结果综合**: 整合信息并生成最终答案

## 工作流程
1. 首先分析用户问题，理解需求
2. 收集必要的信息和数据
3. 使用适当的工具执行操作
4. 整合结果并生成格式化的最终答案

## 答案格式
- 如果需要特定格式，严格按照要求格式化
- 使用 `<answer></answer>` 标签包含最终答案
- 确保答案完整、准确、相关

## 工具使用
- 根据任务需求选择最合适的工具
- 在关键步骤验证结果的正确性
- 对异常结果进行二次确认

请独立完成用户的任务，提供高质量的答案。"""
        
        # 获取所有可用工具
        all_tools = []
        all_tools.extend(self.basic_tools)
        if self.mcp_tools:
            all_tools.extend(self.mcp_tools)
        
        # 创建回退代理
        fallback_agent = Agent(
            model=self.model,
            tools=all_tools,
            system_prompt=fallback_prompt,
            name="fallback_agent"
        )
        
        return fallback_agent
    
    def _handle_critical_error(self, error: Exception, question: str, execution_id: str, start_time: datetime) -> Dict[str, Any]:
        """
        处理关键错误 - 最后的错误处理机制
        
        Args:
            error: 异常对象
            question: 用户问题
            execution_id: 执行ID
            start_time: 开始时间
            
        Returns:
            错误处理结果
        """
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration": duration,
            "question": question[:100],
            "execution_id": execution_id
        }
        
        self.logger.log_error("系统关键错误", error_details)
        
        # 完成性能监控（如果已开始）
        if execution_id:
            try:
                self.performance_monitor.complete_task_execution(
                    execution_id, False, str(error), f"Critical system error: {type(error).__name__}"
                )
            except Exception:
                pass  # 忽略监控记录错误
        
        # 尝试最后的回退策略
        try:
            return self._emergency_fallback(question, error, duration, end_time)
        except Exception as emergency_error:
            # 连紧急回退都失败了
            self.logger.log_error("紧急回退失败", {"emergency_error": str(emergency_error)})
            
            return {
                "success": False,
                "answer": f"系统遇到关键错误无法处理您的请求。错误信息: {error}。请联系系统管理员或稍后重试。",
                "swarm_result": None,
                "duration": duration,
                "agent_path": [],
                "usage": {},
                "timestamp": end_time.isoformat(),
                "execution_mode": "critical_error",
                "error_type": type(error).__name__,
                "error_message": str(error)
            }
    
    def _emergency_fallback(self, question: str, original_error: Exception, duration: float, end_time: datetime) -> Dict[str, Any]:
        """
        紧急回退策略 - 最简单的响应机制
        
        Args:
            question: 用户问题
            original_error: 原始错误
            duration: 执行时长
            end_time: 结束时间
            
        Returns:
            紧急回退结果
        """
        try:
            # 创建最简单的响应
            emergency_response = f"""抱歉，系统在处理您的问题时遇到了技术困难。

问题: {question[:200]}{'...' if len(question) > 200 else ''}

错误类型: {type(original_error).__name__}

建议:
1. 请尝试简化您的问题
2. 检查输入格式是否正确
3. 稍后重试
4. 如果问题持续，请联系技术支持

系统将记录此错误以便改进。"""
            
            return {
                "success": False,
                "answer": emergency_response,
                "swarm_result": None,
                "duration": duration,
                "agent_path": [],
                "usage": {},
                "timestamp": end_time.isoformat(),
                "execution_mode": "emergency_fallback",
                "error_type": type(original_error).__name__,
                "error_message": str(original_error)
            }
            
        except Exception:
            # 连最简单的响应都失败了
            return {
                "success": False,
                "answer": "系统遇到严重错误，无法处理请求。请联系技术支持。",
                "swarm_result": None,
                "duration": duration,
                "agent_path": [],
                "usage": {},
                "timestamp": end_time.isoformat(),
                "execution_mode": "emergency_failed",
                "error_type": "SystemFailure"
            }
    
    def _record_agent_performance(self, agent_path: List[str], duration: float, success: bool, usage: Dict[str, Any]) -> None:
        """记录代理执行性能"""
        try:
            for i, agent_name in enumerate(agent_path):
                # 估算每个代理的执行时间（简化处理）
                agent_duration = duration / len(agent_path) if agent_path else duration
                
                # 提取 token 使用信息
                tokens_consumed = {}
                if usage and isinstance(usage, dict):
                    if 'input_tokens' in usage:
                        tokens_consumed['input_tokens'] = usage['input_tokens'] // len(agent_path) if agent_path else usage['input_tokens']
                    if 'output_tokens' in usage:
                        tokens_consumed['output_tokens'] = usage['output_tokens'] // len(agent_path) if agent_path else usage['output_tokens']
                
                self.performance_monitor.record_agent_execution(
                    agent_name=agent_name,
                    duration=agent_duration,
                    success=success,
                    tools_used=[],  # 工具使用信息需要从 swarm_result 中提取
                    tokens_consumed=tokens_consumed
                )
        except Exception as e:
            self.logger.log_error("记录代理性能失败", {"error": str(e)})
    
    def _record_agent_handoffs(self, agent_path: List[str], success: bool) -> None:
        """记录代理移交"""
        try:
            for i in range(len(agent_path) - 1):
                from_agent = agent_path[i]
                to_agent = agent_path[i + 1]
                self.performance_monitor.record_handoff(
                    from_agent=from_agent,
                    to_agent=to_agent,
                    reason="Task handoff",
                    success=success
                )
                
                # 记录到日志
                self.logger.log_agent_handoff(from_agent, to_agent, "Task handoff")
        except Exception as e:
            self.logger.log_error("记录代理移交失败", {"error": str(e)})
    
    def cleanup(self):
        """清理资源 - 完整的生命周期管理"""
        self.logger.log_info("开始清理 MultiAgentSwarm 资源")
        
        # 清理 Swarm 实例
        try:
            if self.swarm:
                self.logger.log_info("清理 Swarm 实例")
                self.destroy_swarm()
        except Exception as e:
            self.logger.log_error("清理 Swarm 时出错", {"error": str(e)})
        
        # 清理代理列表
        try:
            if self.agents:
                self.logger.log_info(f"清理 {len(self.agents)} 个代理", {"agent_names": [a.name for a in self.agents]})
                self.agents.clear()
        except Exception as e:
            self.logger.log_error("清理代理时出错", {"error": str(e)})
        
        # 清理 MCP 连接
        try:
            mcp_names = [name for name, _ in self.mcp_clients]
            for name, client in self.mcp_clients:
                try:
                    client.stop(None, None, None)
                    self.logger.log_info(f"{name} MCP连接已关闭")
                except Exception as e:
                    self.logger.log_error(f"关闭 {name} MCP连接时出错", {"error": str(e)})
            
            self.mcp_clients.clear()
            self.mcp_tools.clear()
            
            if mcp_names:
                self.logger.log_info("MCP 连接清理完成", {"closed_connections": mcp_names})
                
        except Exception as e:
            self.logger.log_error("清理 MCP 连接时出错", {"error": str(e)})
        
        # 清理基础工具
        try:
            if self.basic_tools:
                tools_count = len(self.basic_tools)
                self.basic_tools.clear()
                self.logger.log_info(f"清理了 {tools_count} 个基础工具")
        except Exception as e:
            self.logger.log_error("清理基础工具时出错", {"error": str(e)})
        
        # 清理性能监控系统
        try:
            if hasattr(self, 'performance_monitor') and self.performance_monitor:
                self.logger.log_info("清理性能监控系统")
                self.performance_monitor.cleanup()
        except Exception as e:
            self.logger.log_error("清理性能监控系统时出错", {"error": str(e)})
        
        # 清理日志系统
        try:
            if hasattr(self, 'logger') and self.logger:
                session_summary = self.logger.get_session_summary()
                self.logger.log_info("MultiAgentSwarm 资源清理完成", session_summary)
                self.logger.cleanup()
        except Exception as e:
        
    
    # 日志相关方法
    def get_logger(self) -> SwarmLogger:
        """获取日志记录器实例"""
        return self.logger
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.logger.get_performance_stats()
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要信息"""
        return self.logger.get_session_summary()
    
    def set_log_level(self, level: str):
        """设置日志级别
        
        Args:
            level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        try:
            log_level = LogLevel(level.upper())
            self.logger.set_log_level(log_level)
        except ValueError:
    
    def enable_strands_debug(self, enable: bool = True):
        """启用或禁用 Strands 官方调试日志
        
        Args:
            enable: 是否启用调试日志
        """
        self.logger.enable_strands_debug = enable
        self.logger._setup_loggers()  # 重新设置日志记录器
        
        status = "启用" if enable else "禁用"
        self.logger.log_info(f"Strands 调试日志已{status}")
    
    def get_log_files(self) -> Dict[str, str]:
        """获取日志文件路径"""
        return self.logger.get_session_summary()["log_files"]
    
    def export_execution_trace(self, output_file: Optional[str] = None) -> str:
        """导出执行跟踪数据
        
        Args:
            output_file: 输出文件路径，如果为None则使用默认路径
            
        Returns:
            导出文件的路径
        """
        if output_file is None:
            output_file = f"execution_trace_export_{self.logger.session_id}.json"
        
        try:
            session_summary = self.get_session_summary()
            performance_stats = self.get_performance_stats()
            
            export_data = {
                "session_info": session_summary,
                "performance_stats": performance_stats,
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.log_info(f"执行跟踪数据已导出", {"output_file": output_file})
            return output_file
            
        except Exception as e:
            self.logger.log_error(f"导出执行跟踪数据失败", {"error": str(e)})
            raise
    
    # 性能监控相关方法
    def get_performance_monitor(self) -> PerformanceMonitor:
        """获取性能监控器实例"""
        return self.performance_monitor
    
    def get_agent_performance_stats(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """获取代理性能统计
        
        Args:
            agent_name: 代理名称，None表示获取所有代理
            
        Returns:
            代理性能统计字典
        """
        if agent_name:
            metrics = self.performance_monitor.get_agent_performance(agent_name)
            if metrics:
                return {
                    "agent_name": metrics.agent_name,
                    "total_executions": metrics.total_executions,
                    "success_rate": metrics.get_success_rate(),
                    "avg_execution_time": metrics.avg_execution_time,
                    "tools_used": dict(metrics.tools_used),
                    "handoffs_initiated": metrics.handoffs_initiated,
                    "handoffs_received": metrics.handoffs_received,
                    "total_tokens": metrics.get_total_tokens(),
                    "error_types": dict(metrics.error_types)
                }
            else:
                return {"error": f"代理 {agent_name} 未找到性能数据"}
        else:
            # 返回所有代理的性能统计
            all_stats = {}
            for name in self.performance_monitor.agent_metrics.keys():
                all_stats[name] = self.get_agent_performance_stats(name)
            return all_stats
    
    def get_tool_performance_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """获取工具性能统计
        
        Args:
            tool_name: 工具名称，None表示获取所有工具
            
        Returns:
            工具性能统计字典
        """
        if tool_name:
            metrics = self.performance_monitor.get_tool_performance(tool_name)
            if metrics:
                return {
                    "tool_name": metrics.tool_name,
                    "total_calls": metrics.total_calls,
                    "success_rate": metrics.get_success_rate(),
                    "avg_execution_time": metrics.avg_execution_time,
                    "agents_using": list(metrics.agents_using),
                    "error_types": dict(metrics.error_types)
                }
            else:
                return {"error": f"工具 {tool_name} 未找到性能数据"}
        else:
            # 返回所有工具的性能统计
            all_stats = {}
            for name in self.performance_monitor.tool_metrics.keys():
                all_stats[name] = self.get_tool_performance_stats(name)
            return all_stats
    
    def get_handoff_patterns(self) -> List[Dict[str, Any]]:
        """获取代理移交模式统计"""
        patterns = self.performance_monitor.get_handoff_patterns()
        return [
            {
                "from_agent": pattern.from_agent,
                "to_agent": pattern.to_agent,
                "count": pattern.count,
                "avg_duration": pattern.avg_duration,
                "success_rate": pattern.success_rate,
                "top_reasons": dict(sorted(pattern.reasons.items(), key=lambda x: x[1], reverse=True)[:3]),
                "last_handoff": pattern.last_handoff_time.isoformat() if pattern.last_handoff_time else None
            }
            for pattern in patterns
        ]
    
    def get_system_health_score(self) -> float:
        """获取系统健康评分 (0-100)"""
        return self.performance_monitor.get_system_health_score()
    
    def get_top_performing_agents(self, metric: str = "success_rate", limit: int = 5) -> List[Tuple[str, float]]:
        """获取表现最佳的代理
        
        Args:
            metric: 评估指标 ('success_rate', 'execution_time', 'tool_usage', 'handoffs')
            limit: 返回数量限制
            
        Returns:
            (代理名, 指标值) 的列表
        """
        return self.performance_monitor.get_top_agents_by_metric(metric, limit)
    
    def get_real_time_performance(self) -> Dict[str, Any]:
        """获取实时性能数据"""
        real_time_stats = self.performance_monitor.get_real_time_stats()
        
        # 添加额外的系统信息
        real_time_stats.update({
            "swarm_status": "active" if self.swarm else "inactive",
            "agents_count": len(self.agents),
            "mcp_tools_count": len(self.mcp_tools),
            "basic_tools_count": len(self.basic_tools),
            "session_id": self.logger.session_id
        })
        
        return real_time_stats
    
    def generate_performance_report(self, 
                                  time_period: Optional[Tuple[datetime, datetime]] = None,
                                  output_file: Optional[str] = None) -> str:
        """生成性能报告
        
        Args:
            time_period: 时间范围，None表示全部时间
            output_file: 输出文件路径，None表示自动生成
            
        Returns:
            报告文件路径
        """
        try:
            report = self.performance_monitor.generate_performance_report(
                time_period=time_period,
                include_recommendations=True
            )
            
            if output_file is None:
                output_file = f"performance_report_{report.report_id}.json"
            
            # 序列化报告
            report_dict = {
                "report_id": report.report_id,
                "generation_time": report.generation_time.isoformat(),
                "time_period": [
                    report.time_period[0].isoformat(),
                    report.time_period[1].isoformat()
                ],
                "summary": report.summary,
                "agent_metrics": {
                    name: {
                        "agent_name": metrics.agent_name,
                        "total_executions": metrics.total_executions,
                        "success_rate": metrics.get_success_rate(),
                        "avg_execution_time": metrics.avg_execution_time,
                        "tools_used": dict(metrics.tools_used),
                        "handoffs": metrics.handoffs_initiated + metrics.handoffs_received,
                        "total_tokens": metrics.get_total_tokens()
                    }
                    for name, metrics in report.agent_metrics.items()
                },
                "tool_metrics": {
                    name: {
                        "tool_name": metrics.tool_name,
                        "total_calls": metrics.total_calls,
                        "success_rate": metrics.get_success_rate(),
                        "avg_execution_time": metrics.avg_execution_time,
                        "agents_using_count": len(metrics.agents_using)
                    }
                    for name, metrics in report.tool_metrics.items()
                },
                "handoff_patterns": [
                    {
                        "from_agent": pattern.from_agent,
                        "to_agent": pattern.to_agent,
                        "count": pattern.count,
                        "success_rate": pattern.success_rate,
                        "avg_duration": pattern.avg_duration
                    }
                    for pattern in report.handoff_patterns
                ],
                "recommendations": report.recommendations
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.log_info(f"性能报告已生成", {"output_file": output_file})
            
            return output_file
            
        except Exception as e:
            self.logger.log_error(f"生成性能报告失败", {"error": str(e)})
            raise
    
    def export_performance_metrics(self, output_file: str, format: str = 'json'):
        """导出性能指标
        
        Args:
            output_file: 输出文件路径
            format: 导出格式 ('json', 'csv')
        """
        try:
            self.performance_monitor.export_metrics(output_file, format)
            self.logger.log_info(f"性能指标已导出", {
                "output_file": output_file,
                "format": format
            })
        except Exception as e:
            self.logger.log_error(f"导出性能指标失败", {"error": str(e)})
            raise
    
    def print_performance_summary(self):
        """打印性能摘要"""
        try:
            real_time_stats = self.get_real_time_performance()
            agent_stats = self.get_agent_performance_stats()
            health_score = self.get_system_health_score()
            
            
            
            for agent_name, stats in agent_stats.items():
                if isinstance(stats, dict) and 'total_executions' in stats:
                          f"成功率 {stats['success_rate']:.1%}, "
                          f"平均耗时 {stats['avg_execution_time']:.2f}s")
            
            # 显示移交模式
            handoff_patterns = self.get_handoff_patterns()
            if handoff_patterns:
                for pattern in sorted(handoff_patterns, key=lambda x: x['count'], reverse=True)[:3]:
                          f"{pattern['count']} 次 (成功率 {pattern['success_rate']:.1%})")
            
            
        except Exception as e:
    
    def enable_performance_monitoring(self, enable: bool = True):
        """启用或禁用性能监控
        
        Args:
            enable: 是否启用性能监控
        """
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.enable_real_time_monitoring = enable
            if enable and not hasattr(self.performance_monitor, 'monitoring_thread'):
                self.performance_monitor._start_real_time_monitoring()
            
            status = "启用" if enable else "禁用"
            self.logger.log_info(f"性能监控已{status}")
        else:
    
    def get_throttling_statistics(self) -> Dict[str, Any]:
        """
        获取限流处理统计信息
        
        Returns:
            限流统计数据
        """
        try:
            if hasattr(self, 'throttling_handler') and self.throttling_handler:
                return self.throttling_handler.get_retry_statistics()
            else:
                return {"error": "限流处理器未初始化"}
        except Exception as e:
            return {"error": f"获取限流统计失败: {e}"}
    
    def configure_throttling(self, 
                           initial_delay: float = None,
                           max_delay: float = None,
                           max_retries: int = None,
                           strategy: str = None):
        """
        配置限流处理参数
        
        Args:
            initial_delay: 初始延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            max_retries: 最大重试次数
            strategy: 重试策略 ('exponential_backoff', 'linear_backoff', 'fixed_delay', 'adaptive')
        """
        try:
            if not hasattr(self, 'throttling_handler') or not self.throttling_handler:
                return
            
            config = self.throttling_handler.config
            
            if initial_delay is not None:
                config.initial_delay = initial_delay
            if max_delay is not None:
                config.max_delay = max_delay
            if max_retries is not None:
                config.max_retries = max_retries
            if strategy is not None:
                try:
                    config.strategy = ThrottlingStrategy(strategy)
                except ValueError:
                    return
            
            
            self.logger.log_info("限流处理配置已更新", {
                "initial_delay": config.initial_delay,
                "max_delay": config.max_delay,
                "max_retries": config.max_retries,
                "strategy": config.strategy.value
            })
            
        except Exception as e:
            self.logger.log_error("配置限流处理失败", {"error": str(e)})
    
    def clear_throttling_history(self):
        """清除限流重试历史记录"""
        try:
            if hasattr(self, 'throttling_handler') and self.throttling_handler:
                self.throttling_handler.clear_history()
                self.logger.log_info("限流重试历史已清除")
            else:
        except Exception as e:
            self.logger.log_error("清除限流历史失败", {"error": str(e)})
    
    def validate_system_configuration(self) -> Dict[str, Any]:
        """
        验证系统配置的完整性 - 实现配置错误的自动修复建议
        
        Returns:
            配置验证结果
        """
        validation_result = {
            "overall_status": "unknown",
            "issues": [],
            "warnings": [],
            "recommendations": [],
            "components": {}
        }
        
        try:
            
            # 验证基础配置
            basic_config_result = self._validate_basic_configuration()
            validation_result["components"]["basic_config"] = basic_config_result
            
            # 验证Swarm配置
            swarm_config_result = self._validate_swarm_configuration()
            validation_result["components"]["swarm_config"] = swarm_config_result
            
            # 验证代理配置
            agent_config_result = self._validate_agent_configuration()
            validation_result["components"]["agent_config"] = agent_config_result
            
            # 验证工具配置
            tool_config_result = self._validate_tool_configuration()
            validation_result["components"]["tool_config"] = tool_config_result
            
            # 验证MCP配置
            mcp_config_result = self._validate_mcp_configuration()
            validation_result["components"]["mcp_config"] = mcp_config_result
            
            # 汇总结果
            all_components = [basic_config_result, swarm_config_result, agent_config_result, 
                            tool_config_result, mcp_config_result]
            
            # 收集所有问题和警告
            for component_result in all_components:
                validation_result["issues"].extend(component_result.get("issues", []))
                validation_result["warnings"].extend(component_result.get("warnings", []))
                validation_result["recommendations"].extend(component_result.get("recommendations", []))
            
            # 确定整体状态
            if any(comp.get("status") == "failed" for comp in all_components):
                validation_result["overall_status"] = "failed"
            elif any(comp.get("status") == "warning" for comp in all_components):
                validation_result["overall_status"] = "warning"
            else:
                validation_result["overall_status"] = "passed"
            
            # 输出验证结果
            self._print_validation_results(validation_result)
            
            return validation_result
            
        except Exception as e:
            error_msg = f"配置验证过程中发生错误: {e}"
            validation_result["overall_status"] = "error"
            validation_result["issues"].append(error_msg)
            
            self.logger.log_error("配置验证失败", {"error": error_msg})
            
            return validation_result
    
    def _validate_basic_configuration(self) -> Dict[str, Any]:
        """验证基础配置"""
        result = {"status": "passed", "issues": [], "warnings": [], "recommendations": []}
        
        # 检查环境变量
        required_env_vars = ["USE_BEDROCK"]
        optional_env_vars = ["SF_API_KEY", "AWS_REGION", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        
        for var in required_env_vars:
            if not os.getenv(var):
                result["issues"].append(f"缺少必需的环境变量: {var}")
                result["status"] = "failed"
        
        for var in optional_env_vars:
            if not os.getenv(var):
                result["warnings"].append(f"可选环境变量未设置: {var}")
                if result["status"] == "passed":
                    result["status"] = "warning"
        
        # 检查模型配置
        try:
            if not self.model:
                result["issues"].append("模型未初始化")
                result["status"] = "failed"
            else:
                model_config = getattr(self.model, 'config', {})
                if not model_config.get('model_id'):
                    result["warnings"].append("模型ID未配置")
                    if result["status"] == "passed":
                        result["status"] = "warning"
        except Exception as e:
            result["issues"].append(f"模型配置检查失败: {e}")
            result["status"] = "failed"
        
        return result
    
    def _validate_swarm_configuration(self) -> Dict[str, Any]:
        """验证Swarm配置"""
        result = {"status": "passed", "issues": [], "warnings": [], "recommendations": []}
        
        # 检查配置文件存在性
        if not os.path.exists(self.config_file):
            result["issues"].append(f"Swarm配置文件不存在: {self.config_file}")
            result["status"] = "failed"
            return result
        
        # 检查配置内容
        if not self.swarm_config:
            result["issues"].append("Swarm配置为空")
            result["status"] = "failed"
            return result
        
        # 检查必需的配置项
        swarm_params = self.swarm_config.get("swarm_config", {})
        required_params = ["max_handoffs", "max_iterations", "execution_timeout", "node_timeout"]
        
        for param in required_params:
            if param not in swarm_params:
                result["issues"].append(f"Swarm配置缺少必需参数: {param}")
                result["status"] = "failed"
        
        # 检查参数合理性
        if "execution_timeout" in swarm_params:
            timeout = swarm_params["execution_timeout"]
            if timeout < 60:
                result["warnings"].append(f"执行超时时间过短: {timeout}秒，建议至少60秒")
                if result["status"] == "passed":
                    result["status"] = "warning"
            elif timeout > 3600:
                result["warnings"].append(f"执行超时时间过长: {timeout}秒，可能影响响应速度")
                if result["status"] == "passed":
                    result["status"] = "warning"
        
        return result
    
    def _validate_agent_configuration(self) -> Dict[str, Any]:
        """验证代理配置"""
        result = {"status": "passed", "issues": [], "warnings": [], "recommendations": []}
        
        # 检查代理是否已创建
        if not hasattr(self, 'agents') or not self.agents:
            result["issues"].append("代理未创建")
            result["status"] = "failed"
            return result
        
        # 检查代理数量
        if len(self.agents) < 4:
            result["issues"].append(f"代理数量不足: 需要4个，实际{len(self.agents)}个")
            result["status"] = "failed"
        
        # 检查必需的代理类型
        required_agents = ["task_analyzer", "info_gatherer", "tool_executor", "result_synthesizer"]
        agent_names = [getattr(agent, 'name', 'unknown') for agent in self.agents]
        
        for required_agent in required_agents:
            if required_agent not in agent_names:
                result["issues"].append(f"缺少必需的代理: {required_agent}")
                result["status"] = "failed"
        
        # 检查代理配置
        for agent in self.agents:
            agent_name = getattr(agent, 'name', 'unknown')
            
            # 检查模型配置
            if not hasattr(agent, 'model') or not agent.model:
                result["issues"].append(f"代理 {agent_name} 缺少模型配置")
                result["status"] = "failed"
            
            # 检查工具配置
            tools_count = 0
            if hasattr(agent, 'tools') and agent.tools:
                tools_count = len(agent.tools)
            elif hasattr(agent, 'tool_registry') and agent.tool_registry:
                try:
                    if hasattr(agent.tool_registry, 'tools'):
                        tools_count = len(agent.tool_registry.tools)
                    elif hasattr(agent.tool_registry, '_tools'):
                        tools_count = len(agent.tool_registry._tools)
                except:
                    tools_count = 0
            
            if tools_count == 0 and agent_name in ["info_gatherer", "tool_executor"]:
                result["warnings"].append(f"代理 {agent_name} 没有配置工具")
                if result["status"] == "passed":
                    result["status"] = "warning"
        
        return result
    
    def _validate_tool_configuration(self) -> Dict[str, Any]:
        """验证工具配置"""
        result = {"status": "passed", "issues": [], "warnings": [], "recommendations": []}
        
        # 检查基础工具
        if not hasattr(self, 'basic_tools') or not self.basic_tools:
            result["issues"].append("基础工具未初始化")
            result["status"] = "failed"
        else:
            basic_tools_count = len(self.basic_tools)
            if basic_tools_count < 3:
                result["warnings"].append(f"基础工具数量较少: {basic_tools_count}个")
                if result["status"] == "passed":
                    result["status"] = "warning"
        
        # 检查MCP工具
        if not hasattr(self, 'mcp_tools'):
            result["warnings"].append("MCP工具未初始化")
            if result["status"] == "passed":
                result["status"] = "warning"
        elif len(self.mcp_tools) == 0:
            result["warnings"].append("没有可用的MCP工具")
            result["recommendations"].append("配置MCP服务器以扩展系统功能")
            if result["status"] == "passed":
                result["status"] = "warning"
        
        return result
    
    def _validate_mcp_configuration(self) -> Dict[str, Any]:
        """验证MCP配置"""
        result = {"status": "passed", "issues": [], "warnings": [], "recommendations": []}
        
        # 检查MCP配置文件
        if not os.path.exists("mcp_config.json"):
            result["warnings"].append("MCP配置文件不存在")
            result["recommendations"].append("创建mcp_config.json以启用MCP功能")
            if result["status"] == "passed":
                result["status"] = "warning"
            return result
        
        # 检查MCP客户端
        if not hasattr(self, 'mcp_clients') or not self.mcp_clients:
            result["warnings"].append("没有活跃的MCP连接")
            result["recommendations"].append("检查MCP服务器配置和连接状态")
            if result["status"] == "passed":
                result["status"] = "warning"
        
        return result
    
    def _print_validation_results(self, validation_result: Dict[str, Any]):
        """打印验证结果"""
        status = validation_result["overall_status"]
        
        
        # 状态图标
        status_icons = {
            "passed": "✅",
            "warning": "⚠️",
            "failed": "❌",
            "error": "💥"
        }
        
        
        # 组件状态
        for component_name, component_result in validation_result["components"].items():
            comp_status = component_result.get("status", "unknown")
            comp_icon = status_icons.get(comp_status, "❓")
        
        # 问题列表
        if validation_result["issues"]:
            for i, issue in enumerate(validation_result["issues"], 1):
        
        # 警告列表
        if validation_result["warnings"]:
            for i, warning in enumerate(validation_result["warnings"], 1):
        
        # 建议列表
        if validation_result["recommendations"]:
            for i, recommendation in enumerate(validation_result["recommendations"], 1):
        
        
        # 根据状态提供总体建议
        if status == "failed":
        elif status == "warning":
        elif status == "passed":
        else:
    
    def auto_fix_configuration(self) -> bool:
        """
        自动修复配置问题 - 实现配置错误的自动修复建议
        
        Returns:
            修复是否成功
        """
        try:
            
            # 先进行配置验证
            validation_result = self.validate_system_configuration()
            
            if validation_result["overall_status"] == "passed":
                return True
            
            fixes_applied = []
            fixes_failed = []
            
            # 修复缺失的配置文件
            if not os.path.exists(self.config_file):
                try:
                    default_config = self._get_default_config()
                    self._save_config(default_config)
                    fixes_applied.append(f"创建默认Swarm配置文件: {self.config_file}")
                except Exception as e:
                    fixes_failed.append(f"创建Swarm配置文件失败: {e}")
            
            # 修复MCP配置
            if not os.path.exists("mcp_config.json"):
                try:
                    self._create_default_mcp_config()
                    fixes_applied.append("创建默认MCP配置文件")
                except Exception as e:
                    fixes_failed.append(f"创建MCP配置文件失败: {e}")
            
            # 重新初始化组件
            try:
                if not hasattr(self, 'basic_tools') or not self.basic_tools:
                    self._setup_basic_tools()
                    fixes_applied.append("重新初始化基础工具")
            except Exception as e:
                fixes_failed.append(f"重新初始化基础工具失败: {e}")
            
            try:
                if not hasattr(self, 'agents') or not self.agents:
                    self.create_specialized_agents()
                    fixes_applied.append("重新创建专业化代理")
            except Exception as e:
                fixes_failed.append(f"重新创建代理失败: {e}")
            
            try:
                if not hasattr(self, 'swarm') or not self.swarm:
                    self.create_swarm()
                    fixes_applied.append("重新创建Swarm实例")
            except Exception as e:
                fixes_failed.append(f"重新创建Swarm失败: {e}")
            
            # 输出修复结果
            
            if fixes_applied:
                for fix in fixes_applied:
            
            if fixes_failed:
                for fix in fixes_failed:
            
            # 再次验证
            final_validation = self.validate_system_configuration()
            
            success = final_validation["overall_status"] in ["passed", "warning"]
            
            if success:
            else:
            
            return success
            
        except Exception as e:
            error_msg = f"自动修复过程中发生错误: {e}"
            self.logger.log_error("自动修复失败", {"error": error_msg})
            return False


def main():
    """多代理系统主入口函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多代理协作系统")
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细模式")
    parser.add_argument("--config", "-c", default="swarm_config.json", help="指定配置文件")
    parser.add_argument("--mode", "-m", choices=["interactive", "single"], default="interactive", 
                       help="运行模式: interactive(交互模式) 或 single(单次问答)")
    parser.add_argument("--question", "-q", help="单次问答模式下的问题")
    parser.add_argument("--system-prompt", "-s", help="自定义系统提示词")
    
    args = parser.parse_args()
    
    
    try:
        # 初始化多代理系统
        swarm_system = MultiAgentSwarm(
            verbose=args.verbose,
            config_file=args.config
        )
        
        
        if args.mode == "single" and args.question:
            # 单次问答模式
            if args.system_prompt:
            
            response = swarm_system.process_question(
                question=args.question,
                system_prompt=args.system_prompt or ""
            )
            
            if response.success:
                
                # 调试信息：检查答案内容
                if hasattr(response, 'swarm_result') and response.swarm_result:
                    if hasattr(response.swarm_result, 'status'):
                
                # 显示答案
                if response.answer and response.answer.strip():
                else:
                    
                    # 如果答案为空，尝试从原始结果中提取更多信息
                    if hasattr(response, 'swarm_result') and response.swarm_result:
                        debug_info = swarm_system._debug_swarm_result_structure(response.swarm_result)
                
            else:
        
        else:
            # 交互模式
            
            while True:
                try:
                    # 获取用户输入
                    question = input("\n💬 你的问题: ").strip()
                    
                    if not question:
                        continue
                    
                    # 处理特殊命令
                    if question.lower() == 'quit':
                        break
                    elif question.lower() == 'help':
                        print_help()
                        continue
                    elif question.lower() == 'verbose':
                        swarm_system.verbose = not swarm_system.verbose
                        continue
                    elif question.lower() == 'status':
                        print_system_status(swarm_system)
                        continue
                    
                    # 获取系统提示词（可选）
                    system_prompt = input("🎯 系统提示词 (可选，直接回车跳过): ").strip()
                    
                    
                    # 处理问题
                    response = swarm_system.process_question(
                        question=question,
                        system_prompt=system_prompt
                    )
                    
                    # 显示结果 - 添加详细的调试信息
                    if response.success:
                        
                        # 调试信息：检查答案内容
                        if swarm_system.verbose:
                            if hasattr(response, 'swarm_result') and response.swarm_result:
                                if hasattr(response.swarm_result, 'status'):
                        
                        # 显示答案
                        if response.answer and response.answer.strip():
                        else:
                            
                            # 如果答案为空，尝试从原始结果中提取更多信息
                            if hasattr(response, 'swarm_result') and response.swarm_result:
                                debug_info = swarm_system._debug_swarm_result_structure(response.swarm_result)
                        
                        if swarm_system.verbose:
                    else:
                
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
    
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    finally:
        # 清理资源
        try:
            if 'swarm_system' in locals():
                swarm_system.cleanup()
        except:
            pass
    
    return 0


def print_help():
    """显示帮助信息"""
    help_text = """
🆘 多代理系统帮助

📋 可用命令:
  quit     - 退出程序
  help     - 显示此帮助信息
  verbose  - 切换详细/简洁模式
  status   - 显示系统状态

💡 使用技巧:
  - 问题要具体明确，如："分析这个Python文件的代码质量"
  - 可以要求使用特定工具，如："用计算器计算复杂表达式"
  - 系统提示词可以定制代理行为，如："你是Python专家"
  - 复杂任务会自动分配给多个专业代理协作完成

🤖 代理类型:
  - 任务分析代理: 分解复杂问题
  - 信息收集代理: 收集和验证信息
  - 工具执行代理: 执行计算和操作
  - 结果综合代理: 整合并格式化答案

🔧 工具能力:
  - 数学计算、时间查询
  - 图像分析、代码执行
  - 网络搜索、JSON处理
  - 浏览器操作等
"""


def print_system_status(swarm_system):
    """显示系统状态"""
    try:
        
        # 显示代理状态
        for agent in swarm_system.agents:
        
        # 显示Swarm状态
        if swarm_system.swarm:
        else:
    
    except Exception as e:


if __name__ == "__main__":
    exit(main())