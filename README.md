# Strands Agent

基于Strands Agents SDK的智能AI代理系统，支持单代理和多代理协作模式，集成MCP工具和多种基础功能。

## 🎯 项目特色

- ✅ **简洁易用**: 专注核心功能，界面清爽
- ✅ **多代理协作**: 支持专业化代理团队协作处理复杂任务
- ✅ **自定义提示词**: 支持任意系统提示词定制
- ✅ **MCP集成**: 支持Model Context Protocol工具
- ✅ **详细模式**: 可查看完整执行过程和代理协作
- ✅ **多种工具**: 数学计算、文件操作、Python执行、网络请求等
- ✅ **性能监控**: 实时监控代理执行性能和资源使用

## 🚀 快速开始

### 1. 创建虚拟环境&安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境

复制环境配置文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的AWS凭证：
```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
```

### 3. 运行Agent

**单代理模式** (传统模式):
```bash
python3 simple_agent.py
```

**多代理协作模式** (推荐用于复杂任务):
```bash
python3 multi_agent_swarm.py
```

## 🤖 多代理协作系统

### 代理专业化分工

多代理模式将复杂任务分解给专业化代理团队：

- **🎯 任务分析代理**: 分析用户需求，制定执行计划
- **📊 信息收集代理**: 收集数据，处理文件和搜索操作
- **🔧 工具执行代理**: 执行计算、代码运行、浏览器操作
- **📝 结果综合代理**: 整合结果，生成最终格式化答案

### 何时使用多代理模式

**推荐使用多代理模式的场景：**
- 复杂的数据分析任务
- 需要多步骤处理的问题
- 涉及多种工具协作的任务
- 需要深度思考和规划的问题

**单代理模式适用场景：**
- 简单的问答
- 单一工具使用
- 快速响应需求

### 多代理配置

编辑 `swarm_config.json` 来自定义多代理行为：

```json
{
  "swarm_config": {
    "max_handoffs": 20,
    "execution_timeout": 900.0,
    "node_timeout": 300.0
  },
  "agents": {
    "task_analyzer": {
      "system_prompt": "自定义任务分析代理提示词...",
      "tools": ["memory"]
    }
  }
}
```

## 💡 选择模式
![alt text](image.png)
```bash
选择模式:
1. 交互模式 
2. 单次问答 (推荐)
请选择 (1-2): 
# 测试时推荐直接填 ‘2’ 即可
```

## 🔍 详细模式

### 简洁模式 vs 详细模式

**简洁模式** (默认):
- 只显示最终回答
- 适合日常使用
- 界面清爽

**详细模式**:
- 显示工具调用过程
- 显示Agent思考步骤
- 显示代理间协作过程 (多代理模式)
- 显示代理切换和任务分配
- 适合调试和学习

### 启用详细模式

**启动时选择:**
```bash
python simple_agent.py
# 选择 'y' 启用详细模式
```

**运行中切换:**
```
💬 你的问题: verbose
# 输入 'verbose' 切换显示模式
```

### 详细模式示例

**单代理模式示例:**
```
🤖 思考中...
🔧 Tool Call: current_time()
📤 Tool Result: 2025-01-20 14:30:25 (Sunday)
🔧 Tool Call: calculator(expression="365-20")
📤 Tool Result: 345
💭 Agent思考: 现在我知道了当前时间和计算结果...
✅ 最终回答: 现在是2025年1月20日14:30，星期日。距离年底还有345天。
```

**多代理协作模式示例:**
```
🤖 启动多代理协作...
👤 [任务分析代理] 分析任务: 需要获取时间并进行计算
🔄 切换到 → [工具执行代理]
🔧 [工具执行代理] Tool Call: current_time()
📤 Tool Result: 2025-01-20 14:30:25 (Sunday)
🔧 [工具执行代理] Tool Call: calculator(expression="365-20")
📤 Tool Result: 345
🔄 切换到 → [结果综合代理]
📝 [结果综合代理] 整合结果并格式化...
✅ 最终回答: 现在是2025年1月20日14:30，星期日。距离年底还有345天。
📊 协作统计: 3个代理参与，2次工具调用，总耗时1.2秒
```

## 🎛️ 交互命令

在交互模式中可以使用以下命令：

- `quit` - 退出程序
- `prompt` - 修改系统提示词
- `verbose` - 切换详细/简洁模式
- `help` - 显示帮助信息

## 🔌 MCP配置

编辑 `mcp_config.json` 来配置MCP服务器：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "npx",
      "args": ["-y", "@smithery/cli@latest", "run", "exa","--key"],
      "disabled": false
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./data.db"],
      "disabled": true
    }
  }
}
```

## 🎯 使用技巧

### 问题设计
- **具体明确**: "用Python分析CSV文件" 比 "分析数据" 更好
- **包含上下文**: 提供必要的背景信息
- **指定格式**: "生成表格"、"画图表"、"写代码"

### 提示词优化
- **明确角色**: 定义专业身份和技能
- **行为指导**: 说明期望的工作方式
- **输出要求**: 指定回答格式和详细程度

### 工具使用
- **组合使用**: 可以要求Agent使用多个工具完成复杂任务
- **文件操作**: 先创建文件，再读取分析
- **图片分析**: 支持分析图片内容、识别文字、描述场景等
- **MCP工具**: 获取实时信息和专业文档

### 图片分析示例
```
问题: "分析这张图片，告诉我图片中有什么内容"
系统提示词: "你是图像分析专家，请详细描述图片内容，包括物体、场景、文字等信息。"

# Agent会自动使用image_reader工具分析图片
```

## 🔍 故障排除

### 常见问题

**Q: MCP工具连接失败？**
```bash
# 检查网络连接
# 验证API密钥
# 确认MCP服务器配置正确
```

**Q: Python代码执行错误？**
```bash
# 检查环境变量设置
export PYTHON_REPL_INTERACTIVE=False
```

**Q: 文件操作权限问题？**
```bash
# 检查目录权限
chmod 755 .
```

**Q: 多代理协作超时？**
```bash
# 检查 swarm_config.json 中的超时设置
# 增加 execution_timeout 和 node_timeout 值
# 确认网络连接稳定
```

**Q: 代理切换过于频繁？**
```bash
# 调整 max_handoffs 参数
# 优化代理的系统提示词
# 检查任务复杂度是否适合多代理模式
```

**Q: 多代理模式性能较慢？**
```bash
# 对于简单任务使用单代理模式
# 调整代理配置减少不必要的工具
# 监控性能报告优化瓶颈
```

### 多代理调试

**启用多代理详细日志：**
```python
# 在 multi_agent_swarm.py 中启用调试
verbose = True  # 显示代理协作过程
```

**查看性能监控：**
```bash
# 运行性能监控脚本
python demo_monitoring.py
```

**分析执行日志：**
```bash
# 查看详细执行日志
ls logs/
# 查看特定会话的日志
cat logs/swarm_session_*.log
```

### 调试模式

启用详细日志：
```python
# 在simple_agent.py开头修改
logging.basicConfig(level=logging.DEBUG)
```

## 📊 性能优化

### 单代理模式优化
- 精简系统提示词
- 明确具体问题
- 合理选择工具

### 多代理模式优化

**选择合适的模式：**
- 简单任务使用单代理模式
- 复杂任务使用多代理模式
- 根据任务类型选择专业代理

**配置优化：**
```json
{
  "swarm_config": {
    "max_handoffs": 15,        // 减少不必要的代理切换
    "execution_timeout": 600,  // 根据任务复杂度调整
    "node_timeout": 180       // 单个代理超时时间
  }
}
```

**代理提示词优化：**
- 明确每个代理的职责边界
- 避免代理功能重叠
- 优化代理间的协作逻辑

**工具分配优化：**
- 只给代理分配必要的工具
- 避免工具冗余配置
- 合理分配计算密集型工具

### 性能监控

**实时监控：**
```bash
# 启用性能监控
python demo_monitoring.py
```

**性能指标：**
- 代理执行时间
- 工具调用次数
- Token 消耗统计
- 代理切换路径

### 控制成本
- 监控Token使用量
- 避免重复查询
- 优化提示词长度
- 合理设置超时时间
- 选择合适的模型

## 🎉 开始使用

1. **安装依赖**: `pip install strands-agents strands-agents-tools mcp`
2. **配置环境**: 编辑 `.env` 文件
3. **选择运行模式**:
   - **单代理模式**: `python simple_agent.py`
   - **多代理模式**: `python multi_agent_swarm.py`
4. **选择交互模式**: 交互模式或单次问答
5. **开始对话**: 输入问题和自定义提示词

### 快速体验多代理协作

```bash
# 启动多代理模式
python multi_agent_swarm.py

# 选择单次问答模式 (输入 2)
# 输入复杂问题，如：
"请分析当前目录下的Python文件，统计代码行数，并生成一个性能报告"

# 观察多个代理如何协作完成任务
```

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**享受与AI代理的智能对话吧！** 🚀
