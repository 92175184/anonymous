# Tool-Call Scaffolding in Tool-Enabled Evaluation

## Overview

The tool-enabled evaluation variant employs a structured tool-call scaffolding mechanism that enables language models to interact with external tools through a standardized interface. This scaffolding provides the infrastructure for models to discover, invoke, and receive results from tools in an iterative, conversational manner. This document explains the architectural foundations, implementation framework, and execution flow of the tool-call scaffolding system.

## Framework Foundation

The tool-call scaffolding is built upon a framework that provides core abstractions for tool integration. The system uses standardized interfaces for tool definition, schema generation, and message passing.

### Core Components

1. **Tool Abstraction**: Tools can be defined as function-based or class-based implementations, providing flexibility for different complexity levels.

2. **Schema Generation**: Tool schemas are automatically generated from function signatures or explicit definitions, enabling structured tool discovery by models.

3. **Message Types**: Structured message types facilitate tool interaction:
   - **AIMessage**: Contains model responses with optional tool call requests
   - **ToolMessage**: Contains tool execution results linked to specific tool calls

## Tool Definition and Registration

### Tool Definition Methods

Tools in the framework are defined using two primary approaches:

**Method 1: Function-Based Tools**

Tools can be defined as simple functions with a decorator:

```python
@tool
async def data_fetch_tool(item_ids: List[str], date: str) -> List[Dict]:
    """Tool description for LLM understanding"""
    # Tool implementation
    pass
```

This approach provides automatic schema inference from type hints and minimal boilerplate.

**Method 2: Class-Based Tools**

For complex tools requiring state management, tools can be defined as classes:

```python
class WebContentTool(BaseTool):
    name: str = "content_fetch"
    description: str = "Tool description"
    args_schema: Type[InputModel] = InputModel

    async def _arun(self, params: List[Params]) -> str:
        # Tool implementation
        pass
```

This approach enables state management, custom initialization, and explicit schema control.

### Tool Schema Generation

When tools are bound to the LLM, their schemas are converted to provider-specific formats. The conversion process extracts tool metadata (name, description, parameters) and transforms them into JSON Schema definitions that include parameter types, required fields, and constraints. These schemas are then adapted to match each LLM provider's function calling API format.

## Tool Binding Mechanism

### Binding Process

Tools are bound to the LLM, creating a runnable instance with tools embedded in its configuration:

```python
llm_with_tools = llm.bind_tools(ALL_TOOLS)
```

This binding process converts tools to the provider's function calling format, creates a bound runnable that includes tools in subsequent API calls, and embeds tool schemas in the request payload.

### API-Level Integration

Tool schemas are included in the LLM API request payload, not just in prompts. This ensures structured tool discovery, provider-native support, and API-level enforcement that only bound tools can be called.

### Configuration Options

The binding mechanism supports configuration for tool choice (required, optional, or disabled), parallel tool calls (multiple tools in one response), and strict schema validation.

## Tool Execution Loop

### Iterative Execution Architecture

The framework implements an iterative tool execution loop that allows models to call tools multiple times before producing a final answer. The loop operates as follows:

1. **Initial Invocation**: The model receives the question and available tool schemas
2. **Tool Call Detection**: The framework checks if the model's response contains `tool_calls`
3. **Tool Execution**: If tool calls are present, each tool is executed and results are collected
4. **Result Integration**: Tool results are added to the conversation history as `ToolMessage` objects
5. **Iteration Continuation**: The model is invoked again with the updated conversation history
6. **Termination**: The loop terminates when the model no longer requests tools or reaches the maximum iteration limit (5 iterations)

### Tool Call Extraction

When the LLM generates a response with tool calls, the response contains structured tool call data:

```python
response.tool_calls = [
    {
        "id": "call_abc123",
        "name": "data_fetch_tool",
        "args": {"item_ids": ["item1"], "date": "2025-01-06"}
    }
]
```

Each tool call includes a unique ID linking to its result, the tool name, and arguments matching the tool's parameter schema.

### Tool Execution

For each tool call, the framework locates the tool, validates arguments, and executes it using async, sync, or direct invocation methods. Errors are caught and returned as structured messages, and tool usage is tracked for analysis.

### Tool Result Integration

After tool execution, results are integrated into the conversation history as `ToolMessage` objects:

```python
ToolMessage(
    content=str(tool_result),
    tool_call_id=tool_call_id,
    name=tool_name
)
```

The `ToolMessage` structure:
- **Content**: The tool execution result (converted to string)
- **Tool call ID**: Links the result to the original tool call request
- **Name**: Tool identifier for reference

This message is appended to the conversation history, enabling the model to see tool results in subsequent iterations.

## API Integration Layer

### Provider Abstraction

The framework abstracts differences between LLM providers through a provider abstraction layer. Different providers implement tool calling through their native APIs, with tools included in requests and tool calls returned in responses. The system supports parallel tool calls and tool choice configuration across providers.

### Schema Conversion

Conversion utilities transform tool schemas to provider-specific formats, adapting schemas to match each provider's requirements and ensuring validity for the target provider.

## Message Flow Architecture

### Conversation State Management

The tool execution loop maintains conversation state through a message list that evolves as follows:

**Initial State**:
```python
messages = [HumanMessage(content=reasoning_prompt)]
```

**After Model Response with Tool Calls**:
```python
messages = [
    HumanMessage(content=reasoning_prompt),
    AIMessage(content="", tool_calls=[...])
]
```

**After Tool Execution**:
```python
messages = [
    HumanMessage(content=reasoning_prompt),
    AIMessage(content="", tool_calls=[...]),
    ToolMessage(content=tool_result_1, tool_call_id="call_1"),
    ToolMessage(content=tool_result_2, tool_call_id="call_2")
]
```

**After Next Model Response**:
```python
messages = [
    HumanMessage(content=reasoning_prompt),
    AIMessage(content="", tool_calls=[...]),
    ToolMessage(...),
    ToolMessage(...),
    AIMessage(content="final_answer")  # No more tool calls
]
```

### Message Type Semantics

Each message type serves a specific role in the tool execution flow:

- **HumanMessage**: Contains user input or system prompts, initiating the conversation
- **AIMessage**: Contains model responses, including optional `tool_calls` when the model requests tool execution
- **ToolMessage**: Contains tool execution results, linked to specific tool calls via `tool_call_id`

This message structure ensures that:
- Tool calls are clearly associated with model requests
- Tool results are properly linked to their originating calls
- The conversation history maintains proper context for the model
