#!/usr/bin/env python3
"""
简洁的Strands Agent - 支持自定义问题和系统提示词
集成MCP工具，单一代理模式
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from strands import Agent, tool
from strands_tools import (
    calculator, current_time, image_reader
)
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient
from strands.hooks import BeforeInvocationEvent
from tools.code_interpreter import AgentCoreCodeInterpreter
from tools.browser import AgentCoreBrowser
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env")


# 配置日志
logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)

USE_BEDROCK=os.getenv("USE_BEDROCK")=="True"
SF_API_KEY=os.getenv("SF_API_KEY")
AWS_REGION=os.getenv("AWS_REGION")

class SimpleAgent:
    """简洁的AI代理"""
    
    def __init__(self, verbose: bool = False, use_bedrock = USE_BEDROCK):
        """
        初始化代理
        
        Args:
            model: 使用的模型名称
            verbose: 是否显示详细执行过程
        """
        if use_bedrock:
            self.model = BedrockModel(
                model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0", 
                region_name=AWS_REGION, 
                temperature=0.7,          
                max_tokens=15000,
                )
        else:
            self.model = OpenAIModel(
                client_args={
                    "api_key": SF_API_KEY,
                    "base_url": "https://api.siliconflow.cn/v1"
                },
                model_id="zai-org/GLM-4.5V",
                params={"max_tokens": 4096, "temperature": 0.7}
                )

        self.verbose = verbose
        self.mcp_clients = []
        self.mcp_tools = []
        
        # built-in工具
        agentcore_code_interpreter = AgentCoreCodeInterpreter(region="us-east-1")
        agentcore_browser = AgentCoreBrowser(region="us-east-1")
        self.basic_tools = [
            #calculator,
            #current_time,
            image_reader,
            agentcore_code_interpreter.code_interpreter,
            #agentcore_browser.browser
        ]
        
        # 尝试连接MCP服务器
        self._setup_mcp()
        
        print(f"Agent初始化完成")
        print(f"Model: {self.model.config['model_id']}")
        print(f"Basic Tools: {len(self.basic_tools)} 个")
        print(f"MCP Tools: {len(self.mcp_tools)} 个")
    

    def _setup_mcp(self):
        """设置MCP连接"""
        try:
            # 读取MCP配置
            if os.path.exists("mcp_config.json"):
                with open("mcp_config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 连接所有启用的服务器
                for name, server_config in config.get("mcpServers", {}).items():
                    if not server_config.get("disabled", False):
                        try:
                            print(f"🔌 连接MCP服务器: {name}")
                            
                            mcp_client = MCPClient(lambda sc=server_config: stdio_client(
                                StdioServerParameters(
                                    command=sc["command"],
                                    args=sc["args"],
                                    env=sc.get("env", {})
                                )
                            ))
                            
                            mcp_client.start()
                            tools = mcp_client.list_tools_sync()
                            
                            self.mcp_clients.append((name, mcp_client))
                            self.mcp_tools.extend(tools)
                            
                            print(f"✅ {name} 连接成功，获得 {len(tools)} 个工具")
                            
                        except Exception as e:
                            print(f"⚠️  MCP服务器 {name} 连接失败: {e}")
                            continue
                
                if self.mcp_tools:
                    print(f"🎯 总计MCP工具: {len(self.mcp_tools)} 个")
                else:
                    print("⚠️  没有成功连接任何MCP服务器")
            else:
                print("⚠️  未找到mcp_config.json，跳过MCP集成")
                
        except Exception as e:
            print(f"⚠️  MCP设置失败: {e}")
    

    def create_agent(self, system_prompt: str) -> Agent:
        """
        创建代理实例
        
        Args:
            system_prompt: 系统提示词
            
        Returns:
            配置好的Agent实例
        """
        all_tools = self.basic_tools + self.mcp_tools
        
        # 根据verbose设置选择回调处理器
        if self.verbose:
            from strands.handlers.callback_handler import PrintingCallbackHandler
            callback_handler = PrintingCallbackHandler()
        else:
            callback_handler = None
 
        agent = Agent(
            model=self.model,
            tools=all_tools,
            system_prompt=system_prompt,
            callback_handler=callback_handler
            )
        
        return agent
    

    def ask(self, question: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        向代理提问 - 改进版本，详细记录和打印 agent 的原始输出
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词，如果为None则使用默认
            
        Returns:
            包含回答和元数据的字典
        """
        if system_prompt is None:
            system_prompt = '''You are an all-capable AI assistant with access to plenty of useful tools, aimed at solving any task presented by the user. ## Task Description:
Please note that the task can be very complex. Do not attempt to solve it all at once. You should break the task down and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest subsequent steps.
Please utilize appropriate tools for the task, then analyze the results obtained from these tools, and provide your reasoning. Always use available tools to verify correctness.
## Workflow:
1. **Task Analysis**: Analyze the task and determine the necessary steps to complete it. Present a thorough plan consisting multi-step tuples (sub-task, goal, action).
2. **Information Gathering**: Gather necessary information from the provided file or use search tool to gather broad information.
3. **Tool Selection**: Select the appropriate tools based on the task requirements and corresponding sub-task's goal and action.
4. **Result Analysis**: Analyze the results obtained from sub-tasks and determine if the original task has been solved.
5. **Final Answer**: If the task has been solved, provide answer in the required format: `<answer>FORMATTED ANSWER</answer>`. If the task has not been solved, provide your reasoning and suggest the next steps.
## Guardrails:
1. Do not use any tools outside of the provided tools list.
2. Always use only one tool at a time in each step of your execution.
3. Even if the task is complex, there is always a solution. 
4. If you can't find the answer using one method, try another approach or use different tools to find the solution.
## Format Requirements:
ALWAYS use the `<answer></answer>` tag to wrap your final answer.
Your `FORMATTED ANSWER` should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. 
- **Number**: If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
- **String**: If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
- **List**: If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
- **Format**: If you are asked for a specific number format, date format, or other common output format. Your answer should be carefully formatted so that it matches the required statement accordingly.
    - `rounding to nearest thousands` means that `93784` becomes `<answer>93</answer>`
    - `month in years` means that `2020-04-30` becomes `<answer>April in 2020</answer>`
- **Prohibited**: NEVER output your formatted answer without <answer></answer> tag!
### Examples
1. <answer>apple tree</answer>
2. <answer>3, 4, 5</answer>
3. <answer>(.*?)</answer>'''
        try:
            start_time = datetime.now()
            
            # 🔍 详细记录执行过程
            print(f"\n🔍 详细分析 Agent 执行过程")
            print("=" * 60)
            print(f"📝 问题: {question}")
            print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 创建代理
            agent = self.create_agent(system_prompt)
            print(f"🤖 Agent 创建完成")
            print(f"   - 模型: {self.model.config['model_id']}")
            print(f"   - 工具数量: {len(self.basic_tools + self.mcp_tools)}")
            
            # 执行查询
            print(f"\n🚀 开始执行查询...")
            response = agent(question)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 🔍 详细分析 Agent 响应结构
            print(f"\n📊 Agent 响应结构分析:")
            self._analyze_response_structure(response)
            
            # 🔍 提取和记录原始输出
            raw_outputs = self._extract_raw_outputs(response)
            
            # 提取响应文本
            answer = self._extract_final_answer(response, raw_outputs)
            
            # 获取使用统计
            usage = self._extract_usage_stats(response)
            
            # 🔍 打印执行摘要
            print(f"\n📋 执行摘要:")
            print(f"   ✅ 成功: True")
            print(f"   📝 答案长度: {len(answer)} 字符")
            print(f"   ⏱️  总耗时: {duration:.2f} 秒")
            print(f"   📊 Token 使用: {usage}")
            print(f"   🔍 原始输出记录: {len(raw_outputs)} 条")
            print("=" * 60)
            
            return {
                "success": True,
                "answer": answer,
                "duration": duration,
                "usage": usage,
                "timestamp": end_time.isoformat(),
                "raw_outputs": raw_outputs,  # 添加原始输出记录
                "response_structure": self._get_response_structure_info(response)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_response_structure(self, response: Any) -> None:
        """
        详细分析 Agent 响应的结构
        
        Args:
            response: Agent 响应对象
        """
        print(f"  📊 响应类型: {type(response).__name__}")
        
        # 检查主要属性
        main_attrs = ['message', 'content', 'text', 'result', 'output', 'metrics', 'usage']
        for attr in main_attrs:
            has_attr = hasattr(response, attr)
            print(f"  - {attr}: {'✅ 存在' if has_attr else '❌ 不存在'}")
            
            if has_attr:
                attr_value = getattr(response, attr)
                if attr_value is not None:
                    attr_type = type(attr_value).__name__
                    if isinstance(attr_value, (str, list, dict)):
                        length = len(attr_value)
                        print(f"    类型: {attr_type}, 长度: {length}")
                    else:
                        print(f"    类型: {attr_type}")
                        
                        # 如果是复杂对象，显示其属性
                        if hasattr(attr_value, '__dict__'):
                            sub_attrs = [sub_attr for sub_attr in dir(attr_value) if not sub_attr.startswith('_')][:3]
                            if sub_attrs:
                                print(f"    子属性: {', '.join(sub_attrs)}")
    
    def _extract_raw_outputs(self, response: Any) -> List[Dict[str, Any]]:
        """
        提取 Agent 的所有原始输出
        
        Args:
            response: Agent 响应对象
            
        Returns:
            原始输出列表
        """
        raw_outputs = []
        
        print(f"\n🔍 提取原始输出:")
        
        # 尝试多种路径提取输出
        extraction_paths = [
            # 直接从 message 提取
            ("response.message", lambda r: r.message if hasattr(r, 'message') else None),
            ("response.content", lambda r: r.content if hasattr(r, 'content') else None),
            ("response.text", lambda r: r.text if hasattr(r, 'text') else None),
            ("response.result", lambda r: r.result if hasattr(r, 'result') else None),
            
            # 从 message 的子属性提取
            ("response.message.content", lambda r: r.message.content if (hasattr(r, 'message') and hasattr(r.message, 'content')) else None),
            ("response.message.text", lambda r: r.message.text if (hasattr(r, 'message') and hasattr(r.message, 'text')) else None),
            
            # 处理字典类型的 message
            ("response.message['content']", lambda r: r.message.get('content') if (hasattr(r, 'message') and isinstance(r.message, dict)) else None),
            
            # 处理列表类型的内容
            ("response.message.content[0].text", self._extract_from_list_content),
            
            # 最后尝试直接转换
            ("str(response)", lambda r: str(r) if r else None)
        ]
        
        for path_name, extractor in extraction_paths:
            try:
                extracted = extractor(response)
                if extracted and str(extracted).strip():
                    output_text = str(extracted).strip()
                    
                    raw_output = {
                        "extraction_path": path_name,
                        "content": output_text,
                        "length": len(output_text),
                        "timestamp": datetime.now().isoformat(),
                        "has_answer_tags": '<answer>' in output_text.lower()
                    }
                    
                    raw_outputs.append(raw_output)
                    
                    print(f"  ✅ {path_name}: {len(output_text)} 字符")
                    print(f"     预览: {output_text[:100]}{'...' if len(output_text) > 100 else ''}")
                    
                    # 检查答案标签
                    if raw_output['has_answer_tags']:
                        print(f"     🎯 包含答案标签")
                        # 提取答案标签内容
                        import re
                        answer_matches = re.findall(r'<answer>(.*?)</answer>', output_text, re.DOTALL | re.IGNORECASE)
                        if answer_matches:
                            answer_content = answer_matches[-1].strip()
                            print(f"     📝 答案内容: {answer_content}")
                else:
                    print(f"  ❌ {path_name}: 无内容")
                    
            except Exception as e:
                print(f"  ⚠️  {path_name}: 提取失败 - {e}")
        
        print(f"\n📊 原始输出汇总: 成功提取 {len(raw_outputs)} 条记录")
        return raw_outputs
    
    def _extract_from_list_content(self, response: Any) -> Optional[str]:
        """
        从列表类型的内容中提取文本
        
        Args:
            response: Agent 响应对象
            
        Returns:
            提取的文本或 None
        """
        try:
            if hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        return first_item['text']
                    elif hasattr(first_item, 'text'):
                        return first_item.text
                    else:
                        return str(first_item)
        except:
            pass
        return None
    
    def _extract_final_answer(self, response: Any, raw_outputs: List[Dict[str, Any]]) -> str:
        """
        从响应和原始输出中提取最终答案
        
        Args:
            response: Agent 响应对象
            raw_outputs: 原始输出列表
            
        Returns:
            最终答案
        """
        print(f"\n🎯 提取最终答案:")
        
        # 优先从包含答案标签的输出中提取
        for output in raw_outputs:
            if output['has_answer_tags']:
                import re
                answer_matches = re.findall(r'<answer>(.*?)</answer>', output['content'], re.DOTALL | re.IGNORECASE)
                if answer_matches:
                    final_answer = answer_matches[-1].strip()
                    print(f"  ✅ 从答案标签提取: {final_answer}")
                    return final_answer
        
        # 如果没有答案标签，使用最长的输出
        if raw_outputs:
            longest_output = max(raw_outputs, key=lambda x: x['length'])
            print(f"  ✅ 使用最长输出: {longest_output['length']} 字符")
            return longest_output['content']
        
        # 最后的回退方案
        fallback_answer = str(response) if response else "无法提取答案"
        print(f"  ⚠️  回退方案: {len(fallback_answer)} 字符")
        return fallback_answer
    
    def _extract_usage_stats(self, response: Any) -> Dict[str, Any]:
        """
        提取使用统计信息
        
        Args:
            response: Agent 响应对象
            
        Returns:
            使用统计字典
        """
        usage = {}
        
        # 尝试多种方式提取使用统计
        if hasattr(response, 'metrics') and response.metrics:
            try:
                usage = response.metrics.accumulated_usage
            except:
                pass
        
        if hasattr(response, 'usage'):
            try:
                usage = response.usage
            except:
                pass
        
        return usage
    
    def _get_response_structure_info(self, response: Any) -> Dict[str, Any]:
        """
        获取响应结构信息
        
        Args:
            response: Agent 响应对象
            
        Returns:
            结构信息字典
        """
        structure_info = {
            "type": type(response).__name__,
            "attributes": []
        }
        
        # 获取所有非私有属性
        for attr in dir(response):
            if not attr.startswith('_'):
                try:
                    value = getattr(response, attr)
                    if not callable(value):
                        structure_info["attributes"].append({
                            "name": attr,
                            "type": type(value).__name__,
                            "has_value": bool(value)
                        })
                except:
                    continue
        
        return structure_info

    def cleanup(self):
        """清理资源"""
        for name, client in self.mcp_clients:
            try:
                client.stop(None, None, None)
                print(f"🧹 {name} MCP连接已关闭")
            except:
                pass


def interactive_mode():
    """交互模式"""
    print("\n🎯 交互模式启动")
    print("输入 'quit' 退出")
    print("输入 'prompt' 修改系统提示词")
    print("输入 'verbose' 切换详细模式")
    print("输入 'help' 查看帮助")
    print("-" * 50)
    
    # 询问是否显示详细过程
    verbose_choice = input("是否显示详细执行过程？(y/n，默认n): ").strip().lower()
    verbose = verbose_choice in ['y', 'yes', '是']
    
    agent = SimpleAgent(verbose=verbose)
    current_prompt = None  # 使用默认提示词
    
    if verbose:
        print("✅ 详细模式已启用 - 将显示工具调用和思考过程")
    else:
        print("ℹ️  简洁模式 - 只显示最终结果")
    
    try:
        while True:
            user_input = input("\n💬 你的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if user_input.lower() == 'prompt':
                print("\n当前系统提示词:")
                if current_prompt:
                    print(current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt)
                else:
                    print("(使用默认提示词)")
                
                new_prompt = input("\n输入新的系统提示词 (回车保持不变): ").strip()
                if new_prompt:
                    current_prompt = new_prompt
                    print("✅ 系统提示词已更新")
                continue
            
            if user_input.lower() == 'verbose':
                agent.cleanup()
                agent.verbose = not agent.verbose
                agent = SimpleAgent(verbose=agent.verbose)
                status = "启用" if agent.verbose else "禁用"
                print(f"✅ 详细模式已{status}")
                continue
            
            if user_input.lower() == 'help':
                show_help()
                continue
            
            if not user_input:
                continue
            
            print("🤖 思考中...")
            result = agent.ask(user_input, current_prompt)
            
            if result["success"]:
                print(f"\n🤖 回答:\n{result['answer']}")
                print(f"\n⏱️  耗时: {result['duration']:.2f}秒")
                if result['usage']:
                    print(f"📊 Token使用: {result['usage']}")
            else:
                print(f"\n❌ 错误: {result['error']}")
    
    except KeyboardInterrupt:
        pass
    finally:
        agent.cleanup()
        print("\n👋 再见！")


def batch_mode():
    """批处理模式"""
    print("\n📝 批处理模式")
    print("请输入你的问题和系统提示词")
    print("-" * 50)
    
    # 获取用户输入
    question = input("💬 你的问题: ").strip()
    if not question:
        print("❌ 问题不能为空")
        return
    
    print("\n📋 系统提示词 (回车使用默认):")
    system_prompt = input().strip()
    if not system_prompt:
        system_prompt = None
    
    # 询问是否显示详细过程
    verbose_choice = input("\n是否显示详细执行过程？(y/n，默认n): ").strip().lower()
    verbose = verbose_choice in ['y', 'yes', '是']
    
    # 执行查询
    agent = SimpleAgent(verbose=verbose)
    
    try:
        print("\n🤖 处理中...")
        result = agent.ask(question, system_prompt)
        
        if result["success"]:
            print(f"\n🤖 回答:\n{result['answer']}")
            print(f"\n📊 统计信息:")
            print(f"   耗时: {result['duration']:.2f}秒")
            if result['usage']:
                print(f"   Token使用: {result['usage']}")
        else:
            print(f"\n❌ 错误: {result['error']}")
    
    finally:
        agent.cleanup()


def main():
    """主函数"""
    print("🚀 Strands Agent")
    print("=" * 30)
    
    print("\n选择模式:")
    print("1. 交互模式 (推荐)")
    print("2. 单次问答")
    
    try:
        choice = input("\n请选择: ").strip()
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            batch_mode()
        else:
            print("无效选择，启动交互模式...")
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 再见！")


if __name__ == "__main__":
    main()