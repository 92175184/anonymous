# Comprehensive AI Agent Evaluation Framework

This repository presents an anonymized implementation of the CAIA evaluation framework, demonstrating the methodology for evaluating Large Language Models (LLMs) both with and without tool access capabilities. The framework is designed for academic research and provides a standardized approach to assess LLM performance on complex reasoning tasks.

## Overview

The CAIA evaluation framework implements a rigorous methodology for comparing LLM responses against ground truth data using automated evaluation techniques. This implementation demonstrates the general evaluation approach used in this research, with certain components anonymized for proprietary reasons while maintaining the core methodological integrity.

**Note**: The tool implementations in this demonstration version are anonymized mock tools that preserve the evaluation methodology without exposing proprietary implementations. In the actual experimental setup, a comprehensive suite of specialized tools provided by various LLM service providers is utilized, along with additional compatible models not included in this public release.

## Repository Structure

- `caia_eval_without_tools.py` - Core evaluator for LLMs without tool access (fully functional)
- `caia_eval_with_tools.py` - Evaluator demonstrating tool-enabled LLM methodology
- `mock_tools.py` - Anonymized tool implementations for methodology demonstration
- `caia_base_evaluator.md` - Standardized evaluation prompt template
- `benchmark.csv` - Research dataset containing 178 evaluation questions
- `requirements.txt` - Python dependency specifications
- `env.example` - Environment configuration template

## Installation and Configuration

### Prerequisites

- Python 3.8 or higher
- Access to LLM API services (OpenAI, Anthropic, Google, etc.)

### Setup Instructions

1. **Environment Setup**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **API Configuration**:
```bash
# Copy the example environment file
cp env.example .env

# Edit .env with your API keys
# Uncomment and fill in the keys for the models you intend to use
```

**Required API Keys** (select based on intended models):
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models  
- `GOOGLE_API_KEY` - For Gemini models

## Usage

### Basic Evaluation Commands

#### Standard LLM Evaluation (Without Tools)
```bash
# Evaluate GPT-4o on a subset of questions
python caia_eval_without_tools.py benchmark --model gpt_4o --limit 5

# Evaluate Claude on 10 questions
python caia_eval_without_tools.py benchmark --model claude_4 --limit 10

# Full evaluation on complete dataset
python caia_eval_without_tools.py benchmark --model gemini_2.5_pro
```

#### Tool-Enabled LLM Evaluation (Methodology Demonstration)
```bash
# Demonstrate tool-enabled evaluation with GPT-4o
python caia_eval_with_tools.py benchmark --model gpt_4o --limit 5

# Tool-enabled evaluation with Claude
python caia_eval_with_tools.py benchmark --model claude_4 --limit 10
```

**Note**: The tool-enabled version utilizes anonymized mock tools to demonstrate the evaluation methodology while preserving proprietary implementations.

### Advanced Configuration Options

#### Concurrency Management
```bash
# Control concurrent API requests (recommended for rate limiting)
python caia_eval_without_tools.py benchmark --model gpt_4o --concurrency 3
python caia_eval_with_tools.py benchmark --model gpt_4o --concurrency 1
```

#### Targeted Evaluation
```bash
# Evaluate specific questions by ID
python caia_eval_without_tools.py benchmark --model gpt_4o --item-id 1
python caia_eval_without_tools.py benchmark --model gpt_4o --item-id 42
```

#### Output Configuration
```bash
# Disable CSV export (console output only)
python caia_eval_without_tools.py benchmark --model gpt_4o --no-export-csv

# Enable detailed logging
python caia_eval_without_tools.py benchmark --model gpt_4o --verbose
```

### Research Use Cases

#### Pilot Studies
```bash
# Quick methodology validation with limited questions
python caia_eval_without_tools.py benchmark --model gpt_4o --limit 3 --concurrency 2
```

#### Comprehensive Evaluation
```bash
# Full dataset evaluation with optimized concurrency
python caia_eval_without_tools.py benchmark --model gpt_4o --concurrency 5
```

#### Comparative Analysis
```bash
# Cross-model performance comparison
python caia_eval_without_tools.py benchmark --model gpt_4o --limit 10
python caia_eval_without_tools.py benchmark --model claude_4 --limit 10
python caia_eval_without_tools.py benchmark --model gemini_2.5_pro --limit 10
```

#### Tool Methodology Demonstration
```bash
# Demonstrate tool-enabled evaluation approach
python caia_eval_with_tools.py benchmark --model gpt_4o --limit 5 --concurrency 2
```

## Evaluation Methodology

### Scoring System

The framework employs a binary scoring system (0.0/1.0) based on automated LLM-based evaluation

### Evaluation Process

1. **Question Processing**: LLM receives standardized question format
2. **Response Generation**: Model generates structured JSON response with reasoning
3. **Automated Judging**: GPT-4 evaluates response against ground truth
4. **Score Assignment**: Binary score with detailed reasoning
5. **Metrics Collection**: Token usage, timing, and performance statistics

### Output Format

#### Console Output Example
```
🎯 Mode: GPT_4O vs Expected Output (using dataset expected_output field)
📊 CSV export: Enabled
INFO:__main__:Found 178 items in CSV dataset
INFO:__main__:Limited to 3 items for sample run
INFO:__main__:🎯 Running GPT_4O evaluation against expected outputs from dataset

📊 Evaluation Summary:
Total comparisons: 3
Final Score Analysis:
  Average score: 0.67
  Min score: 0.0
  Max score: 1.0
Score Distribution:
  Score 0.0: 1 questions (33.3%)
  Score 1.0: 2 questions (66.7%)

🔢 Token Usage Analysis:
Model Token Usage:
  Input tokens: 189 (avg: 63)
  Output tokens: 263 (avg: 88)
  Total tokens: 452 (avg: 151)
```

#### Results Export
```
Exported 3 results to ./tmp_eval_results/gpt_4o_not_tool_enabled_20250923_161423.csv
```

## Dataset Specification

The `benchmark.csv` dataset contains 178 evaluation questions across multiple categories:

| Column | Description |
|--------|-------------|
| `id` | Unique question identifier |
| `question` | Input question text |
| `answer` | Ground truth expected output |
| `category` | Question classification category |

## Supported Models

### Primary Models
- **GPT Series**: `gpt_5`, `gpt_o3`, `gpt_4o`, `gpt_4.1`
- **Claude Series**: `claude_4`, `claude_4.1`
- **Gemini Series**: `gemini_2.5_pro`, `gemini_2.5_flash`

### Additional Models
- **Specialized Models**: `grok_4`, `grok_4_fast`, `kimi_k2`
- **Open Source Models**: `deepseek_r1`, `deepseek_v3p1`, `qwen_3_235b`, `llama_4`
- **Fallback**: Unsupported models default to GPT-4o

## Output Specifications

### CSV Export Format

Results are exported with comprehensive metadata:

**Core Evaluation Data:**
- Question, Model Response, Expected Output
- Final Score (0.0/1.0), Score Reasoning
- Evaluation Timestamp, Experiment ID

**Performance Metrics:**
- Token Usage Statistics (Input/Output/Total)
- Model-specific Performance Data
- Tool Usage Statistics (for tool-enabled evaluations)

**Research Metadata:**
- Question Categories, Item Numbers
- Evaluation Configuration Parameters
- Detailed Reasoning and Analysis

## Implementation Notes

### Anonymization Strategy

This public implementation demonstrates the core evaluation methodology while preserving proprietary components:

1. **Tool Implementations**: Replaced with anonymized mock tools that maintain evaluation flow
2. **Model Configurations**: Simplified to standard LangChain implementations
3. **Evaluation Framework**: Core methodology preserved with anonymized components
4. **Dataset**: Research dataset included for reproducibility

### Research Context

In the actual experimental setup, the following are utilized:
- Comprehensive tool suites from multiple LLM providers
- Additional model variants not included in this public release
- Proprietary evaluation optimizations
- Extended dataset collections

This public release focuses on methodological transparency while maintaining appropriate anonymization boundaries.
