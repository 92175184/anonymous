#!/usr/bin/env python3
"""
CAIA Evaluator With Tools

This script evaluates LLM responses with tool access against expected outputs from the benchmark dataset.
The LLM has access to anonymized crypto analysis tools and decides which ones to use during reasoning.

Usage:
    python caia_eval_with_tools.py benchmark --model gpt_4o
    python caia_eval_with_tools.py benchmark --model claude_4
    python caia_eval_with_tools.py benchmark --model gemini_2.5_pro

    # Available models:
    python caia_eval_with_tools.py benchmark --model gpt_5
    python caia_eval_with_tools.py benchmark --model gpt_o3
    python caia_eval_with_tools.py benchmark --model gpt_4o
    python caia_eval_with_tools.py benchmark --model gpt_4.1
    python caia_eval_with_tools.py benchmark --model grok_4
    python caia_eval_with_tools.py benchmark --model grok_4_fast
    python caia_eval_with_tools.py benchmark --model claude_4
    python caia_eval_with_tools.py benchmark --model claude_4.1
    python caia_eval_with_tools.py benchmark --model gemini_2.5_pro
    python caia_eval_with_tools.py benchmark --model gemini_2.5_flash
    python caia_eval_with_tools.py benchmark --model kimi_k2
    python caia_eval_with_tools.py benchmark --model deepseek_r1
    python caia_eval_with_tools.py benchmark --model deepseek_v3p1
    python caia_eval_with_tools.py benchmark --model qwen_3_235b
    python caia_eval_with_tools.py benchmark --model llama_4

    # Additional options:
    python caia_eval_with_tools.py benchmark --model gpt_4o --limit 10  # Test with 10 items
    python caia_eval_with_tools.py benchmark --model gpt_4o --no-export-csv  # Disable CSV export
    python caia_eval_with_tools.py benchmark --model gpt_4o --concurrency 3  # Limit concurrency
    python caia_eval_with_tools.py benchmark --model gpt_4o --item-id 1  # Run specific item

Note: This version uses anonymized mock tools for demonstration purposes.
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Tool execution imports - anonymized
from mock_tools import ANONYMIZED_TOOLS

# Prompt template paths
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "caia", "prompts")
REASONING_PROMPT_PATH = os.path.join(PROMPTS_DIR, "reasoning_prompt_with_tools.md")
SYNTHESIS_PROMPT_PATH = os.path.join(PROMPTS_DIR, "synthesis_prompt.md")


def load_prompt_template(path: str) -> str:
    """Load a prompt template from a file"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, try to load manually
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Configure logging
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def get_llm_by_model(model_name: str):
    """Get LLM instance by model name"""
    if model_name.startswith("gpt"):
        model_map = {
            "gpt_5": "gpt-4o",  # Fallback to available model
            "gpt_o3": "gpt-4o",
            "gpt_4o": "gpt-4o",
            "gpt_4.1": "gpt-4o",
        }
        return ChatOpenAI(model=model_map.get(model_name, "gpt-4o"), temperature=0)
    elif model_name.startswith("claude"):
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
    elif model_name.startswith("gemini"):
        return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    else:
        # Default to GPT-4 for unsupported models
        return ChatOpenAI(model="gpt-4o", temperature=0)


def _extract_text_from_content(content: Any) -> str:
    """Extract text from various content formats"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Try last block with text
        for block in reversed(content):
            if (
                isinstance(block, dict)
                and "text" in block
                and isinstance(block["text"], str)
            ):
                return block["text"]
        # Fallback: concatenate any text fields
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _parse_structured_response(response_text: str) -> Tuple[str, str]:
    """Parse structured JSON response to extract reasoning and answer"""
    if not response_text:
        return "", ""

    try:
        import json
        import re

        # Clean the response text to extract JSON
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        # Try to find JSON object in the text
        json_match = re.search(
            r'\{[^{}]*"reasoning"[^{}]*"answer"[^{}]*\}', text, re.DOTALL
        )
        if json_match:
            json_text = json_match.group(0)
        else:
            # Fallback: try to parse the entire text as JSON
            json_text = text

        # Parse JSON
        data = json.loads(json_text)

        reasoning = data.get("reasoning", "").strip()
        answer = data.get("answer", "").strip()

        return reasoning, answer

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        _logger.warning(f"Failed to parse structured response: {e}")
        # Fallback to the old parsing method
        return _extract_reasoning_from_response_fallback(response_text)


def _extract_reasoning_from_response_fallback(response_text: str) -> Tuple[str, str]:
    """Fallback method to extract reasoning and final answer from unstructured LLM response"""
    if not response_text:
        return "", ""

    # Common patterns that indicate the start of a final answer
    final_answer_indicators = [
        "Final answer:",
        "Final Answer:",
        "FINAL ANSWER:",
        "Answer:",
        "The answer is:",
        "Based on the data:",
        "Therefore:",
        "In conclusion:",
        "Summary:",
        "Result:",
    ]

    # Look for the last occurrence of any final answer indicator
    reasoning_end = -1
    final_answer_start = -1

    for indicator in final_answer_indicators:
        # Find the last occurrence of this indicator
        last_occurrence = response_text.rfind(indicator)
        if last_occurrence > final_answer_start:
            final_answer_start = last_occurrence
            reasoning_end = last_occurrence

    if final_answer_start != -1:
        # Split the response
        reasoning = response_text[:reasoning_end].strip()
        final_answer = response_text[final_answer_start:].strip()

        # Clean up the final answer by removing the indicator
        for indicator in final_answer_indicators:
            if final_answer.startswith(indicator):
                final_answer = final_answer[len(indicator) :].strip()
                break
    else:
        # No clear final answer indicator found, use full response as reasoning
        reasoning = response_text
        final_answer = ""

    return reasoning, final_answer


def _extract_question_from_input(input_data: Any) -> str:
    """Extract question from various input formats"""
    if isinstance(input_data, str):
        return input_data

    if isinstance(input_data, dict):
        # Handle new format with structured input
        if "new_question" in input_data:
            return str(input_data["new_question"])

    # Fallback: convert to string
    return str(input_data)


def _extract_token_usage(response) -> Dict[str, Any]:
    """Extract token usage from LLM response"""
    token_usage = {}
    try:
        # The response from llm.ainvoke() is typically an AIMessage
        # Check if it has usage_metadata directly
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_data = response.usage_metadata
            token_usage = {
                "input_tokens": usage_data.get("input_tokens", 0),
                "output_tokens": usage_data.get("output_tokens", 0),
                "total_tokens": usage_data.get("total_tokens", 0),
                "cached_tokens": usage_data.get("input_token_details", {}).get(
                    "cache_read", 0
                )
                if isinstance(usage_data.get("input_token_details"), dict)
                else 0,
                "sources_tokens": usage_data.get("num_sources_used", 0),
            }
        else:
            # Try to extract from response_metadata
            if hasattr(response, "response_metadata") and response.response_metadata:
                usage_data = response.response_metadata.get("usage", {})
                if usage_data:
                    token_usage = {
                        "input_tokens": usage_data.get("prompt_tokens", 0),
                        "output_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                        "cached_tokens": 0,  # Not available in response_metadata
                        "sources_tokens": 0,  # Not available in response_metadata
                    }
            else:
                # Fallback: try to convert to LLMResult and use _parse_usage
                from langchain_core.outputs import LLMResult, ChatGeneration

                if hasattr(response, "content"):
                    # It's an AIMessage, wrap it in ChatGeneration and LLMResult
                    generation = ChatGeneration(message=response)
                    llm_result = LLMResult(generations=[[generation]])
                    usage_data = _parse_usage(llm_result) or {}
                    token_usage = {
                        "input_tokens": usage_data.get("input", 0),
                        "output_tokens": usage_data.get("output", 0),
                        "total_tokens": usage_data.get("total", 0),
                        "cached_tokens": usage_data.get("cached", 0),
                        "sources_tokens": usage_data.get("sources", 0),
                    }
                else:
                    # It's already an LLMResult
                    usage_data = _parse_usage(response) or {}
                    token_usage = {
                        "input_tokens": usage_data.get("input", 0),
                        "output_tokens": usage_data.get("output", 0),
                        "total_tokens": usage_data.get("total", 0),
                        "cached_tokens": usage_data.get("cached", 0),
                        "sources_tokens": usage_data.get("sources", 0),
                    }
    except Exception as e:
        _logger.warning(f"Failed to extract token usage: {e}")
        token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "sources_tokens": 0,
        }
    return token_usage


# Anonymized tools list - reduced to hide exact functionality
ALL_TOOLS = ANONYMIZED_TOOLS


def _parse_decision_response(response_text: str) -> Tuple[str, str]:
    """Parse decision-based response to extract decision and reasoning"""
    if not response_text:
        return "tool_call", ""  # Default to tool_call if empty
    
    try:
        import re
        
        # Clean the response text to extract JSON
        text = response_text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        # Try to find JSON object with decision field
        json_match = re.search(
            r'\{[^{}]*"decision"[^{}]*\}', text, re.DOTALL
        )
        if json_match:
            json_text = json_match.group(0)
        else:
            # Fallback: try to parse the entire text as JSON
            json_text = text
        
        # Parse JSON
        data = json.loads(json_text)
        
        decision = data.get("decision", "tool_call").strip().lower()
        reasoning = data.get("reasoning", "").strip()
        
        # Normalize decision value
        if decision not in ["tool_call", "synthesis"]:
            decision = "tool_call"  # Default to tool_call for safety
        
        return decision, reasoning
    
    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        _logger.warning(f"Failed to parse decision response: {e}")
        # Default to tool_call to continue the loop
        return "tool_call", response_text


async def run_tool_enabled_llm(
    question: str, expected_output: str, model_name: str
) -> Tuple[str, str, str, str, str, Dict[str, Any], Dict[str, Any]]:
    """Run question through LLM with access to all tools using agentic workflow"""
    try:
        # Get LLM
        llm = get_llm_by_model(model_name)

        # Load external prompt templates
        reasoning_template = load_prompt_template(REASONING_PROMPT_PATH)
        synthesis_template = load_prompt_template(SYNTHESIS_PROMPT_PATH)

        # Create tools introduction for LLM decision-making
        tools_intro = """
**Available Tool Categories:**

**Crypto Data Tools:**
- crypto_data_tool_1: General crypto market data and metrics
- crypto_data_tool_2: Historical price and volume analysis
- crypto_data_tool_3: Token and project information

**Market Analysis Tools:**
- market_analysis_tool_1: Technical analysis and indicators
- market_analysis_tool_2: Market trends and patterns

**Blockchain Data Tools:**
- blockchain_data_tool_1: On-chain transaction analysis
- blockchain_data_tool_2: Smart contract data retrieval
- blockchain_data_tool_3: Blockchain metrics and statistics

**Social & Research Tools:**
- social_sentiment_tool: Social media sentiment analysis
- search_tool: Real-time information search
- web_crawl_tool: Web content extraction
- code_execution_tool: Computational analysis
"""

        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        # Run LLM with tools in an agentic loop
        max_iterations = 5
        iteration = 0
        all_tool_results = []
        tool_results_text = "No tool results yet."
        initial_reasoning = ""  # Store reasoning from iterations

        # Token usage tracking
        total_token_usage = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cached_tokens": 0,
            "total_sources_tokens": 0,
            "iteration_breakdown": [],
        }

        # Enhanced tool usage tracking
        tool_usage_stats = {
            "total_tool_calls": 0,
            "unique_tools_used": set(),
            "tool_call_list": [],
            "tools_by_iteration": {},
        }

        # Build initial reasoning prompt
        initial_prompt = reasoning_template.format(
            question=question,
            tools_intro=tools_intro,
            tool_results_text=tool_results_text,
        )
        messages = [HumanMessage(content=initial_prompt)]

        # Agentic decision-based loop
        while iteration < max_iterations:
            iteration += 1
            _logger.info(f"  🔄 LLM iteration {iteration}/{max_iterations}")

            response = await llm_with_tools.ainvoke(messages)

            # Add response to conversation history
            messages.append(response)

            # Extract token usage from response
            iteration_token_usage = _extract_token_usage(response)
            total_token_usage["iteration_breakdown"].append(iteration_token_usage)

            # Update total token usage
            total_token_usage["total_input_tokens"] += iteration_token_usage.get(
                "input_tokens", 0
            )
            total_token_usage["total_output_tokens"] += iteration_token_usage.get(
                "output_tokens", 0
            )
            total_token_usage["total_tokens"] += iteration_token_usage.get(
                "total_tokens", 0
            )
            total_token_usage["total_cached_tokens"] += iteration_token_usage.get(
                "cached_tokens", 0
            )
            total_token_usage["total_sources_tokens"] += iteration_token_usage.get(
                "sources_tokens", 0
            )

            # Extract response text and parse decision
            response_text = _extract_text_from_content(
                getattr(response, "content", "")
            )
            decision, reasoning = _parse_decision_response(response_text)

            # Accumulate reasoning from each iteration
            if reasoning:
                if initial_reasoning:
                    initial_reasoning += f"\n\n[Iteration {iteration}]: {reasoning}"
                else:
                    initial_reasoning = f"[Iteration {iteration}]: {reasoning}"

            _logger.info(f"  📋 Decision: {decision}")

            # Check if model decided to proceed to synthesis
            if decision == "synthesis":
                _logger.info(
                    f"  ✅ LLM decided to proceed to synthesis after {iteration} iterations"
                )
                break

            # Check if the LLM wants to call tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                _logger.info(
                    f"  🛠️  LLM wants to call {len(response.tool_calls)} tools: {[call['name'] for call in response.tool_calls]}"
                )

                # Execute the tools
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]

                    _logger.info(f"    ⚡ Executing tool: {tool_name}")

                    # Update tool usage tracking
                    tool_usage_stats["total_tool_calls"] += 1
                    tool_usage_stats["unique_tools_used"].add(tool_name)
                    tool_usage_stats["tool_call_list"].append(
                        {
                            "iteration": iteration,
                            "tool_name": tool_name,
                            "arguments": tool_args,
                            "success": True,
                        }
                    )

                    # Track tools by iteration
                    if iteration not in tool_usage_stats["tools_by_iteration"]:
                        tool_usage_stats["tools_by_iteration"][iteration] = []
                    tool_usage_stats["tools_by_iteration"][iteration].append(tool_name)

                    # Find and execute the tool
                    tool_result = "Tool execution failed"
                    for tool in ALL_TOOLS:
                        if tool.name == tool_name:
                            try:
                                if hasattr(tool, "ainvoke"):
                                    tool_result = await tool.ainvoke(tool_args)
                                elif hasattr(tool, "invoke"):
                                    tool_result = tool.invoke(tool_args)
                                else:
                                    tool_result = str(tool(**tool_args))
                                break
                            except Exception as e:
                                tool_result = f"Tool execution error: {str(e)}"
                                _logger.error(f"Tool {tool_name} execution failed: {e}")

                                # Update tool usage tracking for failed calls
                                tool_usage_stats["tool_call_list"][-1]["success"] = (
                                    False
                                )

                    # Add tool result to conversation history
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                    )

                    # Store tool result for accumulation
                    all_tool_results.append(f"{tool_name}: {str(tool_result)}")

                # Update tool_results_text for next iteration
                tool_results_text = "\n".join(all_tool_results)

                # Add follow-up prompt for next decision
                follow_up_prompt = f"""Tool execution complete. You have gathered the following results so far:

{tool_results_text}

Based on the original question: "{question}"

Decide whether you need more information (call more tools) or have enough to synthesize an answer.

Respond with a JSON object:
{{
  "decision": "tool_call" or "synthesis",
  "reasoning": "Your reasoning here..."
}}"""
                messages.append(HumanMessage(content=follow_up_prompt))
            else:
                # No tool calls and decision wasn't synthesis - break to avoid infinite loop
                _logger.info(
                    f"  ✅ LLM completed without tool calls after {iteration} iterations"
                )
                break

        # Now synthesize all tool results into a final answer
        if all_tool_results:
            _logger.info(f"  🔄 Synthesizing {len(all_tool_results)} tool results")

            # Create synthesis prompt using external template
            synthesis_content = synthesis_template.format(
                question=question,
                initial_reasoning=initial_reasoning,
                tool_results_text=tool_results_text,
            )
            synthesis_prompt = HumanMessage(content=synthesis_content)

            # Get final synthesized answer
            final_response = await llm_with_tools.ainvoke([synthesis_prompt])

            # Extract token usage from final response
            final_token_usage = _extract_token_usage(final_response)
            total_token_usage["iteration_breakdown"].append(final_token_usage)

            # Update total token usage
            total_token_usage["total_input_tokens"] += final_token_usage.get(
                "input_tokens", 0
            )
            total_token_usage["total_output_tokens"] += final_token_usage.get(
                "output_tokens", 0
            )
            total_token_usage["total_tokens"] += final_token_usage.get(
                "total_tokens", 0
            )
            total_token_usage["total_cached_tokens"] += final_token_usage.get(
                "cached_tokens", 0
            )
            total_token_usage["total_sources_tokens"] += final_token_usage.get(
                "sources_tokens", 0
            )

            llm_response = _extract_text_from_content(
                getattr(final_response, "content", "")
            )
            _logger.info(f"  ✅ Synthesis complete")
        else:
            # No tools were used, use the original response
            llm_response = _extract_text_from_content(getattr(response, "content", ""))
            _logger.info(f"  ℹ️  No tools used, using direct LLM response")

        # Extract reasoning and final answer from the structured LLM response
        reasoning, final_answer = _parse_structured_response(llm_response)

        # If we couldn't separate them well, use the full response as reasoning
        if not reasoning:
            reasoning = llm_response

        # Check for empty response and log warning
        if not llm_response or llm_response.strip() == "":
            _logger.warning(
                f"Empty LLM response detected for question: {question[:100]}..."
            )
            _logger.warning(
                "Consider running check_empty_responses.py to rerun empty items"
            )

        # Extract tool names and parameters used for tracking
        tools_used = []
        tool_params_used = []

        for tool_result in all_tool_results:
            if ":" in tool_result:
                tool_name = tool_result.split(":")[0]
                tools_used.append(tool_name)
            else:
                tools_used.append(tool_result)

        # Extract tool parameters from messages (ToolMessage objects)
        for message in messages:
            if hasattr(message, "name") and hasattr(message, "tool_call_id"):
                # This is a ToolMessage - find the corresponding tool call
                tool_name = message.name
                tool_call_id = message.tool_call_id

                # Find the original tool call in messages
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if tool_call.get("id") == tool_call_id:
                                tool_params_used.append(
                                    f"{tool_name}({tool_call.get('args', {})})"
                                )
                                break

        # Finalize tool usage stats
        tool_usage_stats["unique_tools_used"] = list(
            tool_usage_stats["unique_tools_used"]
        )
        tool_usage_stats["unique_tools_count"] = len(
            tool_usage_stats["unique_tools_used"]
        )

        # Create tool usage summary
        tools_used_str = ", ".join(tools_used) if tools_used else "None"
        tool_params_str = " | ".join(tool_params_used) if tool_params_used else "None"
        num_tools_used = len(tools_used)

        # For tool results, we'll use the concatenated tool results
        tool_results = reasoning if all_tool_results else "No tools were used"

        _logger.info(
            f"  ✅ Item processing complete - Used {num_tools_used} tools: {tools_used_str}"
        )
        return (
            llm_response,
            tool_results,
            reasoning,
            final_answer,
            tools_used_str,
            num_tools_used,
            tool_params_str,
            total_token_usage,
            tool_usage_stats,
        )

    except Exception as e:
        _logger.error(f"Tool-enabled {model_name.upper()} execution failed: {e}")
        return (
            f"{model_name.upper()} Error: {str(e)}",
            None,
            "",
            "",
            "",
            "None",
            0,
            "None",
            {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_cached_tokens": 0,
                "total_sources_tokens": 0,
                "iteration_breakdown": [],
            },
            {
                "total_tool_calls": 0,
                "unique_tools_used": [],
                "unique_tools_count": 0,
                "tool_call_list": [],
                "tools_by_iteration": {},
            },
        )


def get_comparison_prompt(
    question: str, llm_response: str, expected_output: str, current_date: str
) -> str:
    """Get comparison prompt for LLM response vs expected output evaluation"""
    # Load the prompt template from caia_base_evaluator.md
    template_path = os.path.join(os.path.dirname(__file__), "caia_base_evaluator.md")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Replace template variables
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{ai_response}}", llm_response)
    prompt = prompt.replace("{{expected_output}}", expected_output)
    prompt = prompt.replace("{{current_date}}", current_date)
    
    return prompt


async def llm_judge_comparison(
    question: str, llm_response: str, expected_output: str, current_date: str
) -> dict:
    """Use LLM to judge and compare LLM response against expected output"""

    prompt = get_comparison_prompt(
        question, llm_response, expected_output, current_date
    )

    try:
        # Use GPT-4 for judging
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        messages = [HumanMessage(content=prompt)]

        response = await llm.ainvoke(messages)

        # Extract and clean content
        content = _extract_text_from_content(getattr(response, "content", ""))

        # Clean JSON response
        text = content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            result_data = json.loads(text)
            return result_data
        except json.JSONDecodeError:
            _logger.error(
                f"Failed to parse LLM response as JSON. Content: {text[:200]}..."
            )
            raise

    except Exception as e:
        _logger.error(f"LLM judgment failed: {e}")
        return {
            "final_score": "0.0",
        }


async def process_dataset_item(
    item, experiment_id: str, model_name: str, item_index: int, total_items: int
) -> Optional[Dict[str, Any]]:
    """Process a single dataset item comparing tool-enabled LLM response against expected output"""
    try:
        # Get the question from the dataset item input
        raw_input = getattr(item, "input", "") or ""
        if not raw_input:
            _logger.error(f"No input found in dataset item")
            return None

        # Extract question from potentially structured input
        question = _extract_question_from_input(raw_input)
        if not question:
            _logger.error(f"No question found in dataset item input")
            return None

        # Get expected output from dataset
        expected_output = getattr(item, "expected_output", None)
        if not expected_output:
            _logger.warning(f"No expected output found in dataset item, skipping")
            return None

        # Get item metadata to extract item_number
        metadata = getattr(item, "metadata", {}) or {}
        item_number = metadata.get("item_number", None)

        # Log progress
        _logger.info(
            f"🔄 Processing item {item_index + 1}/{total_items}: {question[:100]}..."
        )

        # Run tool-enabled LLM
        (
            llm_response,
            tool_results,
            reasoning,
            final_answer,
            tools_used_str,
            num_tools_used,
            tool_params_str,
            total_token_usage,
            tool_usage_stats,
        ) = await run_tool_enabled_llm(question, expected_output, model_name)

        # Compare using LLM judge
        current_date = datetime.now().strftime("%Y-%m-%d")
        eval_result = await llm_judge_comparison(
            question, llm_response, expected_output, current_date
        )

        # Create result
        result = {
            "question": question,
            "llm_response": llm_response,
            "final_answer": final_answer,
            "expected_output": expected_output,
            "final_score": eval_result.get("final_score", "0.0"),
            "score_reason": eval_result.get("score_reason", ""),
            "evaluated_at": datetime.now().isoformat(),
            "llm_judge": "caia_base_judge",
            "experiment_id": experiment_id,
            "model": model_name,
            "tool_results": tool_results,
            "tool_enabled": True,
            "reasoning": reasoning,
            "tools_used": tools_used_str,
            "num_tools_used": num_tools_used,
            "tool_params": tool_params_str,
            "item_number": item_number,
            # Token usage data
            "total_input_tokens": total_token_usage.get("total_input_tokens", 0),
            "total_output_tokens": total_token_usage.get("total_output_tokens", 0),
            "total_tokens": total_token_usage.get("total_tokens", 0),
            "total_cached_tokens": total_token_usage.get("total_cached_tokens", 0),
            "total_sources_tokens": total_token_usage.get(
                "total_sources_tokens", 0
            ),
            # Enhanced tool usage data
            "total_tool_calls": tool_usage_stats.get("total_tool_calls", 0),
            "unique_tools_count": tool_usage_stats.get("unique_tools_count", 0),
            "unique_tools_used": ", ".join(
                tool_usage_stats.get("unique_tools_used", [])
            ),
            "tool_call_list": tool_usage_stats.get("tool_call_list", []),
            "tools_by_iteration": tool_usage_stats.get("tools_by_iteration", {}),
        }

        return result

    except Exception as e:
        _logger.exception(f"Failed to process item: {e}")
        return None


def print_evaluation_statistics(results: List[Dict[str, Any]]) -> None:
    """Print simple statistics about evaluation results"""
    if not results:
        print("No results to analyze")
        return

    print(f"\n📊 Tool-Enabled LLM Evaluation Summary:")
    print(f"Total comparisons: {len(results)}")

    # Calculate final score statistics
    scores = []
    num_tools_per_question = []
    tool_usage_count = {}

    for result in results:
        try:
            score = float(result.get("final_score", "0.0"))
            scores.append(score)
        except (ValueError, TypeError):
            continue

        # Track tool usage
        num_tools = result.get("num_tools_used", 0)
        num_tools_per_question.append(num_tools)

        tools_used = result.get("tools_used", "")
        if tools_used and tools_used != "None":
            for tool in tools_used.split(", "):
                tool_usage_count[tool] = tool_usage_count.get(tool, 0) + 1

    if scores:
        print(f"\nFinal Score Analysis:")
        print(f"  Average score: {sum(scores) / len(scores):.2f}")
        print(f"  Min score: {min(scores):.1f}")
        print(f"  Max score: {max(scores):.1f}")

        # Score distribution
        score_0 = len([s for s in scores if s == 0.0])
        score_1 = len([s for s in scores if s == 1.0])
        print(f"\nScore Distribution:")
        print(f"  Score 0.0: {score_0} questions ({score_0 / len(scores) * 100:.1f}%)")
        print(f"  Score 1.0: {score_1} questions ({score_1 / len(scores) * 100:.1f}%)")
    else:
        print("No valid scores found")

    # Tool usage statistics
    if num_tools_per_question:
        print(f"\n🛠️  Tool Usage Analysis:")
        print(
            f"  Average tools per question: {sum(num_tools_per_question) / len(num_tools_per_question):.1f}"
        )
        print(f"  Min tools used: {min(num_tools_per_question)}")
        print(f"  Max tools used: {max(num_tools_per_question)}")

        # Questions with no tools
        no_tools = len([n for n in num_tools_per_question if n == 0])
        print(
            f"  Questions with no tools: {no_tools} ({no_tools / len(num_tools_per_question) * 100:.1f}%)"
        )

    # Most used tools
    if tool_usage_count:
        print(f"\n🔝 Most Used Tools:")
        sorted_tools = sorted(
            tool_usage_count.items(), key=lambda x: x[1], reverse=True
        )
        for tool, count in sorted_tools[:10]:  # Top 10 tools
            percentage = count / len(results) * 100
            print(f"  {tool}: {count} times ({percentage:.1f}%)")

    # Token usage statistics
    total_input_tokens = [r.get("total_input_tokens", 0) for r in results]
    total_output_tokens = [r.get("total_output_tokens", 0) for r in results]
    total_tokens = [r.get("total_tokens", 0) for r in results]
    total_cached_tokens = [r.get("total_cached_tokens", 0) for r in results]
    total_sources_tokens = [r.get("total_sources_tokens", 0) for r in results]

    print(f"\n🔢 Token Usage Analysis:")
    print(
        f"  Total input tokens: {sum(total_input_tokens):,} (avg: {sum(total_input_tokens) / len(results):.0f})"
    )
    print(
        f"  Total output tokens: {sum(total_output_tokens):,} (avg: {sum(total_output_tokens) / len(results):.0f})"
    )
    print(
        f"  Total tokens: {sum(total_tokens):,} (avg: {sum(total_tokens) / len(results):.0f})"
    )
    if sum(total_cached_tokens) > 0:
        print(f"  Cached tokens: {sum(total_cached_tokens):,}")
    if sum(total_sources_tokens) > 0:
        print(f"  Sources tokens: {sum(total_sources_tokens):,}")

    # Check for empty responses
    empty_responses = [r for r in results if not r.get("llm_response", "").strip()]
    if empty_responses:
        print(f"\n⚠️  Empty Response Detection:")
        print(
            f"  Found {len(empty_responses)} empty responses out of {len(results)} total"
        )
        print(
            f"  Empty response rate: {len(empty_responses) / len(results) * 100:.1f}%"
        )
        print(
            f"  💡 Run 'python check_empty_responses.py --rerun-empty --model {results[0].get('model', 'MODEL_NAME')}' to fix empty responses"
        )


def export_results_to_csv(
    results: List[Dict[str, Any]], dataset_name: str, model_name: str
) -> Optional[str]:
    """Export results to CSV with simplified columns"""
    if not results:
        return None

    try:
        # Generate filename and ensure output directory exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_tool_enabled_{timestamp}.csv"
        output_dir = os.path.join(os.path.dirname(__file__), "tmp_eval_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Write CSV using standard library
        fieldnames = [
            "Question",
            "Llm_response",
            "Final_answer",
            "Expected_output",
            "Final_score",
            "Score_reason",
            "Evaluated_at",
            "Llm_judge",
            "Experiment_id",
            "Model",
            "Tool_results",
            "Tool_enabled",
            "Reasoning",
            "Tools_used",
            "Num_tools_used",
            "Tool_params",
            "Item_number",
            # Token usage columns
            "Total_input_tokens",
            "Total_output_tokens",
            "Total_tokens",
            "Total_cached_tokens",
            "Total_sources_tokens",
            # Enhanced tool usage columns
            "Total_tool_calls",
            "Unique_tools_count",
            "Unique_tools_used",
            "Tool_call_list",
            "Tools_by_iteration",
        ]
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                lineterminator="\n",
            )
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "Question": r.get("question", ""),
                        "Llm_response": r.get("llm_response", ""),
                        "Final_answer": r.get("final_answer", ""),
                        "Expected_output": r.get("expected_output", ""),
                        "Final_score": r.get("final_score", "0.0"),
                        "Score_reason": r.get("score_reason", ""),
                        "Evaluated_at": r.get("evaluated_at", ""),
                        "Llm_judge": r.get("llm_judge", ""),
                        "Experiment_id": r.get("experiment_id", ""),
                        "Model": r.get("model", ""),
                        "Tool_results": r.get("tool_results", ""),
                        "Tool_enabled": r.get("tool_enabled", ""),
                        "Reasoning": r.get("reasoning", ""),
                        "Tools_used": r.get("tools_used", ""),
                        "Num_tools_used": r.get("num_tools_used", 0),
                        "Tool_params": r.get("tool_params", ""),
                        "Item_number": r.get("item_number", ""),
                        # Token usage data
                        "Total_input_tokens": r.get("total_input_tokens", 0),
                        "Total_output_tokens": r.get("total_output_tokens", 0),
                        "Total_tokens": r.get("total_tokens", 0),
                        "Total_cached_tokens": r.get("total_cached_tokens", 0),
                        "Total_sources_tokens": r.get("total_sources_tokens", 0),
                        # Enhanced tool usage data
                        "Total_tool_calls": r.get("total_tool_calls", 0),
                        "Unique_tools_count": r.get("unique_tools_count", 0),
                        "Unique_tools_used": r.get("unique_tools_used", ""),
                        "Tool_call_list": json.dumps(r.get("tool_call_list", [])),
                        "Tools_by_iteration": json.dumps(
                            r.get("tools_by_iteration", {})
                        ),
                    }
                )

        _logger.info("📊 Exported %d results to %s", len(results), output_path)
        return filename

    except Exception as e:
        _logger.error("Failed to export CSV: %s", e)
        return None


async def evaluate_tool_enabled_llm_vs_expected(
    dataset_name: str,
    model_name: str,
    max_concurrency: int,
    item_id: str | None = None,
    export_csv: bool = False,
    limit: int | None = None,
) -> None:
    """Main evaluation function for tool-enabled LLM vs expected output comparison"""
    try:
        # Load dataset from CSV file
        csv_path = os.path.join(os.path.dirname(__file__), "benchmark.csv")
        if not os.path.exists(csv_path):
            raise ValueError(f"CSV file '{csv_path}' not found")
        
        df = pd.read_csv(csv_path)
        _logger.info(f"Found {len(df)} items in CSV dataset")
        
        # Convert DataFrame to list of items with expected attributes
        items = []
        for _, row in df.iterrows():
            item = type('DatasetItem', (), {
                'input': row['question'],
                'expected_output': row['answer'],
                'metadata': {'item_number': row['id'], 'category': row['category']},
                'id': str(row['id'])
            })()
            items.append(item)

        # Filter by item_id if specified
        if item_id:
            items = [item for item in items if getattr(item, "id", "") == item_id]
            _logger.info(f"Filtered to {len(items)} items matching item_id: {item_id}")

        # Limit number of items if specified
        if limit:
            items = items[:limit]
            _logger.info(f"Limited to {len(items)} items for sample run")

        # Generate experiment ID
        experiment_id = f"{model_name}_tool_enabled_vs_expected_{datetime.now().strftime('%m%d_%H%M%S')}"

        _logger.info(
            f"🎯 Running tool-enabled {model_name.upper()} evaluation against expected outputs from dataset"
        )

        # Process items with concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)
        results = []

        async def process_with_semaphore(item, index):
            async with semaphore:
                return await process_dataset_item(
                    item, experiment_id, model_name, index, len(items)
                )

        tasks = [process_with_semaphore(item, i) for i, item in enumerate(items)]

        # Process with progress tracking
        completed = 0
        results = []

        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
                completed += 1
                _logger.info(
                    f"📈 Progress: {completed}/{len(items)} items completed ({completed / len(items) * 100:.1f}%)"
                )

        _logger.info(f"Successfully processed {len(results)} items")

        # Print statistics
        print_evaluation_statistics(results)

        # Export results if requested
        if export_csv and results:
            export_results_to_csv(results, dataset_name, model_name)

        # Evaluation completed

        print(
            f"✅ Tool-enabled {model_name.upper()} vs Expected Output comparison completed!"
        )

    except Exception as e:
        _logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CAIA Evaluator With Tools"
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Langfuse dataset name to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "gpt_5",
            "gpt_o3",
            "gpt_4o",
            "gpt_4.1",
            "grok_4",
            "grok_4_fast",
            "claude_4",
            "claude_4.1",
            "gemini_2.5_pro",
            "gemini_2.5_flash",
            "kimi_k2",
            "gpt_oss_120b",
            "deepseek_r1",
            "deepseek_v3p1",
            "qwen_3_235b",
            "llama_4",
        ],
        default="gpt_5",
        help="Model to use for evaluation (gpt_5, gpt_o3, gpt_4o, gpt_4.1, grok_4, grok_4_fast, claude_4, claude_4.1, gemini_2.5_pro, gemini_2.5_flash, kimi_k2, gpt_oss_120b, deepseek_r1, deepseek_v3p1, qwen_3_235b, or llama_4)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=24, help="Max concurrent comparisons"
    )
    parser.add_argument("--item-id", type=str, help="Run only specific item ID")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export results to CSV (default: True)",
    )
    parser.add_argument(
        "--no-export-csv", action="store_true", help="Disable CSV export"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of items to process (for sample runs)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle CSV export logic
    export_csv = args.export_csv and not args.no_export_csv

    print(
        f"🎯 Mode: Tool-enabled {args.model.upper()} vs Expected Output (using dataset expected_output field)"
    )
    if export_csv:
        print("📊 CSV export: Enabled")
    else:
        print("📊 CSV export: Disabled")

    asyncio.run(
        evaluate_tool_enabled_llm_vs_expected(
            dataset_name=args.dataset,
            model_name=args.model,
            max_concurrency=args.concurrency,
            item_id=args.item_id,
            export_csv=export_csv,
            limit=args.limit,
        )
    )
