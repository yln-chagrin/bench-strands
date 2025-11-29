# Strands Agents -- Swarm

An intelligent AI agent system built on Strands Agents SDK, supporting both single-agent and multi-agent collaboration modes with MCP integration.

## Features

- Single and multi-agent collaboration modes
- Custom system prompts
- Model Context Protocol (MCP) integration
- Verbose mode for debugging
- Built-in tools: calculator, file operations, Python execution, web requests
- Performance monitoring

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Run

**Single Agent Mode:**
```bash
python3 simple_agent.py
```

**Multi-Agent Mode** (for complex tasks):
```bash
python3 multi_agent_swarm.py
```

## Multi-Agent System

The multi-agent mode uses specialized agents:

- **Task Analyzer**: Analyzes requirements and creates execution plans
- **Information Collector**: Handles data collection and file operations
- **Tool Executor**: Executes calculations and code
- **Result Synthesizer**: Integrates results and formats final answers

### When to Use Multi-Agent Mode

**Use multi-agent for:**
- Complex data analysis
- Multi-step processing
- Tasks requiring multiple tools
- Problems needing deep planning

**Use single-agent for:**
- Simple Q&A
- Single tool usage
- Quick responses

## Configuration

### Multi-Agent Config

Edit `swarm_config.json`:

```json
{
  "swarm_config": {
    "max_handoffs": 20,
    "execution_timeout": 900.0,
    "node_timeout": 300.0
  }
}
```

### MCP Config

Edit `mcp_config.json`:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "npx",
      "args": ["-y", "@smithery/cli@latest", "run", "exa","--key"],
      "disabled": false
    }
  }
}
```

## Interactive Commands

- `quit` - Exit program
- `prompt` - Modify system prompt
- `verbose` - Toggle verbose/concise mode
- `help` - Show help

## Troubleshooting

**MCP connection failed:**
- Check network connection
- Verify API keys
- Confirm MCP server configuration

**Multi-agent timeout:**
- Adjust timeout settings in `swarm_config.json`
- Check network stability

**Performance issues:**
- Use single-agent for simple tasks
- Optimize agent configuration
- Monitor performance reports

## License

MIT License
