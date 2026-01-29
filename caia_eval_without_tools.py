#!/usr/bin/env python3
"""
CAIA Evaluator Without Tools

This script evaluates LLM responses without tool access against expected outputs from the benchmark dataset.
The LLM answers questions directly using only its training data and reasoning capabilities.

Usage:
    python caia_eval_without_tools.py benchmark --model gpt_4o
    python caia_eval_without_tools.py benchmark --model claude_4
    python caia_eval_without_tools.py benchmark --model gemini_2.5_pro

    # Available models:
    python caia_eval_without_tools.py benchmark --model gpt_5
    python caia_eval_without_tools.py benchmark --model gpt_o3
    python caia_eval_without_tools.py benchmark --model gpt_4o
    python caia_eval_without_tools.py benchmark --model gpt_4.1
    python caia_eval_without_tools.py benchmark --model grok_4
    python caia_eval_without_tools.py benchmark --model grok_4_fast
    python caia_eval_without_tools.py benchmark --model claude_4
    python caia_eval_without_tools.py benchmark --model gemini_2.5_pro
    python caia_eval_without_tools.py benchmark --model gemini_2.5_flash
    python caia_eval_without_tools.py benchmark --model kimi_k2
    python caia_eval_without_tools.py benchmark --model deepseek_r1
    python caia_eval_without_tools.py benchmark --model deepseek_v3p1
    python caia_eval_without_tools.py benchmark --model qwen_3_235b
    python caia_eval_without_tools.py benchmark --model llama_4

    # Additional options:
    python caia_eval_without_tools.py benchmark --model gpt_4o --limit 10  # Test with 10 items
    python caia_eval_without_tools.py benchmark --model gpt_4o --no-export-csv  # Disable CSV export
    python caia_eval_without_tools.py benchmark --model gpt_4o --concurrency 3  # Limit concurrency
    python caia_eval_without_tools.py benchmark --model gpt_4o --item-id 1  # Run specific item
"""

import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

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
        return ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
    else:
        # Default to GPT-4 for unsupported models
        return ChatOpenAI(model="gpt-4o", temperature=0)


async def run_model_ask(
    question: str, expected_output: str, model_name: str
) -> Tuple[str, Optional[str], str, Dict[str, Any]]:
    """Run question through specified model with direct question input"""
    try:
        # Get LLM
        llm = get_llm_by_model(model_name)

        # Create a prompt that asks for reasoning in JSON format
        reasoning_prompt = f"""Please answer the following question and explain your reasoning process step by step.

Question: {question}

Please provide your response in the following JSON format:
{{
    "answer": "Your direct answer to the question",
    "reasoning": "Your step-by-step reasoning process explaining how you arrived at this answer, what steps you took, and what information you considered"
}}

Respond with valid JSON only."""

        # Run LLM call with reasoning prompt
        response = await llm.ainvoke([HumanMessage(content=reasoning_prompt)])

        # Extract response text
        model_text = _extract_text_from_content(getattr(response, "content", ""))

        # Simple token usage tracking (mock values for anonymized version)
        token_usage = {
            "input_tokens": len(reasoning_prompt.split()),
            "output_tokens": len(model_text.split()),
            "total_tokens": len(reasoning_prompt.split()) + len(model_text.split()),
            "cached_tokens": 0,
            "sources_tokens": 0,
        }

        # Extract reasoning from JSON response
        reasoning = ""
        answer = model_text

        try:
            # Clean JSON response
            json_text = model_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            # Parse JSON
            parsed_response = json.loads(json_text)
            answer = parsed_response.get("answer", model_text)
            reasoning = parsed_response.get("reasoning", "")

        except (json.JSONDecodeError, KeyError) as e:
            _logger.warning(
                f"Failed to parse JSON response, using full text as reasoning: {e}"
            )
            # If JSON parsing fails, use the full response as reasoning
            reasoning = model_text

        # No trace ID needed in anonymized version
        model_trace_id = f"mock_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return answer, model_trace_id, reasoning, token_usage

    except Exception as e:
        _logger.error(f"{model_name.upper()} execution failed: {e}")
        return f"{model_name.upper()} Error: {str(e)}", None, "", {}


def get_comparison_prompt(
    question: str, model_response: str, expected_output: str, current_date: str
) -> str:
    """Get comparison prompt for model vs expected output evaluation"""
    # Load the prompt template from caia_base_evaluator.md
    template_path = os.path.join(os.path.dirname(__file__), "caia_base_evaluator.md")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    # Replace template variables
    prompt = template.replace("{{question}}", question)
    prompt = prompt.replace("{{ai_response}}", model_response)
    prompt = prompt.replace("{{expected_output}}", expected_output)
    prompt = prompt.replace("{{current_date}}", current_date)
    
    return prompt


async def llm_judge_comparison(
    question: str, model_response: str, expected_output: str, current_date: str
) -> Tuple[dict, Dict[str, Any]]:
    """Use LLM to judge and compare model response against expected output"""

    prompt = get_comparison_prompt(
        question, model_response, expected_output, current_date
    )

    try:
        # Use GPT-4 for judging
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

        messages = [HumanMessage(content=prompt)]

        response = await llm.ainvoke(messages)

        # Extract and clean content
        content = _extract_text_from_content(getattr(response, "content", ""))

        # Simple token usage tracking for judge
        judge_token_usage = {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(content.split()),
            "total_tokens": len(prompt.split()) + len(content.split()),
            "cached_tokens": 0,
            "sources_tokens": 0,
        }

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
            return result_data, judge_token_usage
        except json.JSONDecodeError:
            _logger.error(
                f"Failed to parse LLM response as JSON. Content: {text[:200]}..."
            )
            raise

    except Exception as e:
        _logger.error(f"LLM judgment failed: {e}")
        return {
            "final_score": "0.0",
        }, {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "sources_tokens": 0,
        }


async def process_dataset_item(
    item,
    experiment_id: str,
    model_name: str,
    item_index: int = 0,
    total_items: int = 0,
) -> Optional[Dict[str, Any]]:
    """Process a single dataset item comparing model response against expected output"""
    try:
        # Get the question from the dataset item input
        question = getattr(item, "input", "") or ""
        if not question:
            _logger.error(f"No input question found in dataset item")
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

        # Run model
        (
            model_response,
            model_trace_id,
            reasoning,
            model_token_usage,
        ) = await run_model_ask(question, expected_output, model_name)

        # Compare using LLM judge
        current_date = datetime.now().strftime("%Y-%m-%d")
        eval_result, judge_token_usage = await llm_judge_comparison(
            question, model_response, expected_output, current_date
        )

        # Create result
        result = {
            "question": question,
            "model_response": model_response,
            "expected_output": expected_output,
            "final_score": eval_result.get("final_score", "0.0"),
            "score_reason": eval_result.get("score_reason", ""),
            "model_trace_id": model_trace_id or "unknown",
            "evaluated_at": datetime.now().isoformat(),
            "llm_judge": "llm_judge",
            "experiment_id": experiment_id,
            "model": model_name,
            "reasoning": reasoning,
            "item_number": item_number,
            # Token usage for model
            "model_input_tokens": model_token_usage.get("input_tokens", 0),
            "model_output_tokens": model_token_usage.get("output_tokens", 0),
            "model_total_tokens": model_token_usage.get("total_tokens", 0),
            "model_cached_tokens": model_token_usage.get("cached_tokens", 0),
            "model_sources_tokens": model_token_usage.get("sources_tokens", 0),
            # Token usage for judge
            "judge_input_tokens": judge_token_usage.get("input_tokens", 0),
            "judge_output_tokens": judge_token_usage.get("output_tokens", 0),
            "judge_total_tokens": judge_token_usage.get("total_tokens", 0),
            "judge_cached_tokens": judge_token_usage.get("cached_tokens", 0),
            "judge_sources_tokens": judge_token_usage.get("sources_tokens", 0),
            # Combined totals
            "total_input_tokens": model_token_usage.get("input_tokens", 0)
            + judge_token_usage.get("input_tokens", 0),
            "total_output_tokens": model_token_usage.get("output_tokens", 0)
            + judge_token_usage.get("output_tokens", 0),
            "total_tokens": model_token_usage.get("total_tokens", 0)
            + judge_token_usage.get("total_tokens", 0),
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

    print(f"\n📊 Evaluation Summary:")
    print(f"Total comparisons: {len(results)}")

    # Calculate final score statistics
    scores = []
    for result in results:
        try:
            score = float(result.get("final_score", "0.0"))
            scores.append(score)
        except (ValueError, TypeError):
            continue

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

    # Calculate token usage statistics
    model_input_tokens = [r.get("model_input_tokens", 0) for r in results]
    model_output_tokens = [r.get("model_output_tokens", 0) for r in results]
    model_total_tokens = [r.get("model_total_tokens", 0) for r in results]
    judge_input_tokens = [r.get("judge_input_tokens", 0) for r in results]
    judge_output_tokens = [r.get("judge_output_tokens", 0) for r in results]
    judge_total_tokens = [r.get("judge_total_tokens", 0) for r in results]
    total_input_tokens = [r.get("total_input_tokens", 0) for r in results]
    total_output_tokens = [r.get("total_output_tokens", 0) for r in results]
    total_tokens = [r.get("total_tokens", 0) for r in results]

    print(f"\n🔢 Token Usage Analysis:")
    print(f"Model Token Usage:")
    print(
        f"  Input tokens: {sum(model_input_tokens):,} (avg: {sum(model_input_tokens) / len(results):.0f})"
    )
    print(
        f"  Output tokens: {sum(model_output_tokens):,} (avg: {sum(model_output_tokens) / len(results):.0f})"
    )
    print(
        f"  Total tokens: {sum(model_total_tokens):,} (avg: {sum(model_total_tokens) / len(results):.0f})"
    )

    print(f"\nJudge Token Usage:")
    print(
        f"  Input tokens: {sum(judge_input_tokens):,} (avg: {sum(judge_input_tokens) / len(results):.0f})"
    )
    print(
        f"  Output tokens: {sum(judge_output_tokens):,} (avg: {sum(judge_output_tokens) / len(results):.0f})"
    )
    print(
        f"  Total tokens: {sum(judge_total_tokens):,} (avg: {sum(judge_total_tokens) / len(results):.0f})"
    )

    print(f"\nCombined Token Usage:")
    print(
        f"  Total input tokens: {sum(total_input_tokens):,} (avg: {sum(total_input_tokens) / len(results):.0f})"
    )
    print(
        f"  Total output tokens: {sum(total_output_tokens):,} (avg: {sum(total_output_tokens) / len(results):.0f})"
    )
    print(
        f"  Total tokens: {sum(total_tokens):,} (avg: {sum(total_tokens) / len(results):.0f})"
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
        filename = f"{model_name}_not_tool_enabled_{timestamp}.csv"
        output_dir = os.path.join(os.path.dirname(__file__), "tmp_eval_results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Write CSV using standard library
        fieldnames = [
            "Question",
            "Model_response",
            "Expected_output",
            "Final_score",
            "Score_reason",
            "Model_trace_id",
            "Evaluated_at",
            "Llm_judge",
            "Experiment_id",
            "Model",
            "Reasoning",
            "Item_number",
            # Model token usage
            "Model_input_tokens",
            "Model_output_tokens",
            "Model_total_tokens",
            "Model_cached_tokens",
            "Model_sources_tokens",
            # Judge token usage
            "Judge_input_tokens",
            "Judge_output_tokens",
            "Judge_total_tokens",
            "Judge_cached_tokens",
            "Judge_sources_tokens",
            # Combined totals
            "Total_input_tokens",
            "Total_output_tokens",
            "Total_tokens",
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
                        "Model_response": r.get("model_response", ""),
                        "Expected_output": r.get("expected_output", ""),
                        "Final_score": r.get("final_score", "0.0"),
                        "Score_reason": r.get("score_reason", ""),
                        "Model_trace_id": r.get("model_trace_id", ""),
                        "Evaluated_at": r.get("evaluated_at", ""),
                        "Llm_judge": r.get("llm_judge", ""),
                        "Experiment_id": r.get("experiment_id", ""),
                        "Model": r.get("model", ""),
                        "Reasoning": r.get("reasoning", ""),
                        "Item_number": r.get("item_number", ""),
                        # Model token usage
                        "Model_input_tokens": r.get("model_input_tokens", 0),
                        "Model_output_tokens": r.get("model_output_tokens", 0),
                        "Model_total_tokens": r.get("model_total_tokens", 0),
                        "Model_cached_tokens": r.get("model_cached_tokens", 0),
                        "Model_sources_tokens": r.get("model_sources_tokens", 0),
                        # Judge token usage
                        "Judge_input_tokens": r.get("judge_input_tokens", 0),
                        "Judge_output_tokens": r.get("judge_output_tokens", 0),
                        "Judge_total_tokens": r.get("judge_total_tokens", 0),
                        "Judge_cached_tokens": r.get("judge_cached_tokens", 0),
                        "Judge_sources_tokens": r.get("judge_sources_tokens", 0),
                        # Combined totals
                        "Total_input_tokens": r.get("total_input_tokens", 0),
                        "Total_output_tokens": r.get("total_output_tokens", 0),
                        "Total_tokens": r.get("total_tokens", 0),
                    }
                )

        _logger.info("📊 Exported %d results to %s", len(results), output_path)
        return filename

    except Exception as e:
        _logger.error("Failed to export CSV: %s", e)
        return None


async def evaluate_model_vs_expected(
    dataset_name: str,
    model_name: str,
    max_concurrency: int = 5,
    item_id: str | None = None,
    export_csv: bool = False,
    limit: int | None = None,
) -> None:
    """Main evaluation function for model vs expected output comparison"""
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
        experiment_id = (
            f"{model_name}_vs_expected_{datetime.now().strftime('%m%d_%H%M%S')}"
        )

        _logger.info(
            f"🎯 Running {model_name.upper()} evaluation against expected outputs from dataset"
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

        print(f"✅ {model_name.upper()} vs Expected Output comparison completed!")

    except Exception as e:
        _logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CAIA Ask Evaluator - Model vs Expected Output Comparison"
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
        help="Model to use for evaluation (gpt_5, gpt_o3, gpt_4o, gpt_4.1, grok_4, grok_4_fast, claude_4, gemini_2.5_pro, gemini_2.5_flash, kimi_k2, gpt_oss_120b, deepseek_r1, deepseek_v3p1, qwen_3_235b, or llama_4)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=6, help="Max concurrent comparisons"
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
        f"🎯 Mode: {args.model.upper()} vs Expected Output (using dataset expected_output field)"
    )
    if export_csv:
        print("📊 CSV export: Enabled")
    else:
        print("📊 CSV export: Disabled")

    asyncio.run(
        evaluate_model_vs_expected(
            dataset_name=args.dataset,
            model_name=args.model,
            max_concurrency=args.concurrency,
            item_id=args.item_id,
            export_csv=export_csv,
            limit=args.limit,
        )
    )
