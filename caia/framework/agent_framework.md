# CAIA Framework: Methodology and Architecture

## Introduction

The CAIA Ask Evaluator Framework constitutes a systematic evaluation methodology designed to assess language model performance on factual question-answering tasks. The framework implements a rigorous comparative methodology that evaluates model responses against ground-truth expected outputs derived from curated datasets, utilizing an LLM-based judge to determine response correctness. This document presents the framework's methodological foundations, architectural design principles, and evaluation procedures, providing researchers with the necessary information to understand, replicate, and extend the evaluation methodology.

## Framework Architecture

The evaluation framework is structured around three primary phases: (1) pre-evaluation setup and configuration, (2) model response generation and evaluation, and (3) post-evaluation analysis and reporting. Each phase comprises multiple interconnected components that collectively ensure systematic, reproducible evaluation procedures.

### Phase 1: Pre-Evaluation Setup

#### Dataset Preparation

The framework processes evaluation datasets comprising question-answer pairs with ground-truth expected outputs. Each dataset item contains a question to be answered and the corresponding expected answer for comparative evaluation. The framework implements flexible dataset filtering mechanisms to facilitate targeted evaluation of specific subsets, enabling focused analysis and iterative refinement of evaluation procedures.

#### Model Configuration

The framework accommodates evaluation across diverse language model providers and architectural variants. Models are configured using consistent evaluation parameters while accommodating each model's unique capabilities and operational requirements, thereby ensuring equitable and comparable assessments across heterogeneous architectures.

#### Prompt Design

The framework utilizes structured prompts that mandate models to provide both answers and explicit reasoning in standardized formats. This design facilitates analysis of reasoning quality and ensures consistent, parseable responses that support automated evaluation and comparative analysis.

#### Experiment Management

Each evaluation run is assigned a unique identifier that facilitates traceability, reproducibility, and comparative analysis across different evaluation configurations, enabling systematic tracking and replication of experimental conditions.

### Phase 2: Model Response Generation and Evaluation

#### Model Response Generation

Model response generation constitutes the core component of the evaluation framework. The process transforms questions into structured model outputs through carefully designed prompts and iterative reasoning mechanisms.

**Non-Tool Response Generation**

For models without tool access, the framework implements a direct question-answering methodology:

1. **Prompt Construction**: Questions are embedded within structured reasoning prompts that require both answers and explicit reasoning in JSON format.

2. **Model Invocation**: Models process the prompt and generate responses containing:
   - A direct answer to the question
   - Step-by-step reasoning explaining the derivation of the answer

3. **Response Parsing**: The framework extracts and separates answers from reasoning processes, accommodating various response formats and edge cases through robust parsing mechanisms.

This approach facilitates evaluation of intrinsic model knowledge and reasoning capabilities without external augmentation.

**Tool-Enabled Response Generation**

The tool-enabled variant implements an agentic workflow that decouples information gathering from answer synthesis:

1. **Iterative Tool Selection**: Models analyze questions and strategically select tools to gather required information. During each iteration, models:
   - Review previously gathered tool results
   - Determine whether to continue tool calls or proceed to synthesis
   - Select appropriate tools based on information requirements

2. **Tool Execution Loop**: Models execute tools across multiple iterations (maximum of 5), enabling:
   - Incremental information gathering
   - Adaptive strategy formulation based on intermediate results
   - Multi-source information integration

3. **Synthesis Phase**: Following tool execution completion, models synthesize all gathered information into comprehensive final answers. This synthesis process:
   - Integrates initial reasoning with tool results
   - Produces structured outputs with calculations and specific values
   - Ensures coherent integration of all information sources

**Key Design Principles**

- **Separation of Concerns**: Tool selection is decoupled from answer generation, enabling focused decision-making during information gathering phases
- **Structured Output**: All responses utilize standardized JSON formats, ensuring parseable and consistent outputs
- **Iterative Refinement**: Models refine their understanding through multiple reasoning steps and tool invocations

#### LLM-Based Evaluation Judge

Following model response generation, the framework employs a separate LLM judge to compare model answers against expected outputs. This design choice addresses several methodological considerations:

- **Semantic Understanding**: LLM judges assess semantic equivalence and partial correctness more effectively than exact string matching algorithms
- **Contextual Evaluation**: The judge incorporates context, nuance, and partial alignment that simple metrics may fail to capture
- **Scalability**: Automated LLM-based judging scales to large evaluation datasets without requiring human evaluators, enabling efficient processing of extensive evaluation corpora



## Tool-Enabled Evaluator Variant

The framework incorporates a complementary variant that evaluates models with tool access, enabling assessment of how models leverage external tools to answer questions. This variant provides insights into tool selection strategies, execution efficiency, and synthesis capabilities.

### Architectural Differences

The tool-enabled variant adheres to a similar three-phase structure while incorporating two key architectural enhancements:

1. **Iterative Tool Execution Loop**: Models invoke tools multiple times (up to 5 iterations) before producing final answers, enabling multi-step reasoning with external data sources.

2. **Synthesis Step**: Following tool execution completion, the framework performs a synthesis step that integrates initial reasoning with all tool results to produce final answers.

### Available Tools

The tool-enabled evaluator provides access to 23 comprehensive tools across eight categories:

**Market Data Tools** (4 tools): Historical price data, market rankings, trader positioning signals, and volatility alerts.

**Technical Analysis Tools** (1 tool): Momentum and trend indicators for market analysis.

**Discovery/Search Tools** (3 tools): Web search, URL content extraction, and trending topic detection.

**Social Sentiment Tools** (2 tools): Social media search and curated discussion streams.

**DeFi Metrics Tools** (5 tools): Protocol engagement metrics, revenue analysis, network activity summaries, and exchange overviews.

**On-Chain Data Tools** (6 tools): Supply queries, ledger data queries, time conversion, balance queries, transaction lookups, and block information.

**Token Analytics Tools** (1 tool): Release schedule and vesting information.

**Execution Tools** (1 tool): Code execution workspace for calculations and comparisons.

### Tool Execution Methodology

The tool-enabled variant implements an iterative execution mechanism that enables models to:

1. Analyze questions and available tool results
2. Determine which tools to invoke (if any) based on question requirements
3. Execute selected tools and receive results
4. Continue iterating or produce final answers

The loop terminates when models no longer request tool calls or reach the maximum iteration limit (5 iterations). This design facilitates:
- Incremental information gathering
- Understanding refinement through multiple tool calls
- Information integration from multiple sources
- Strategy adaptation based on intermediate results

### Synthesis Process

Following tool execution completion, the framework performs a synthesis step that integrates initial reasoning with all tool results. The synthesis prompt instructs models to:

- Build upon initial analysis
- Incorporate tool results effectively
- Directly address the question posed
- Include relevant numerical data and calculations
- Provide specific values, percentages, or ratios where applicable
- Synthesize all information into coherent final answers

This synthesis step ensures that final answers incorporate all gathered information rather than relying solely on tool outputs, thereby promoting coherent and comprehensive responses.

### Prompt Design for Tool-Enabled Evaluation

The tool-enabled variant utilizes specialized prompts that guide models through the agentic workflow:

**Reasoning Prompt**: Facilitates iterative tool selection by:
- Presenting available tools and their capabilities
- Requiring models to analyze information requirements
- Instructing models to make explicit decisions regarding tool call continuation or synthesis progression
- Emphasizing strategic tool selection over exhaustive execution

**Synthesis Prompt**: Following tool execution, guides final answer generation by:
- Integrating initial reasoning with all tool results
- Requiring synthesis rather than repetition of raw tool data
- Ensuring complete answers with calculations and specific values

This two-stage prompt design decouples information gathering from answer generation, enabling models to focus on strategic tool selection during iterations and comprehensive synthesis for final answers.

### Evaluation Consistency

The tool-enabled variant utilizes the same evaluation judge prompt as the non-tool version, ensuring consistent scoring criteria across both evaluation modes. This consistency facilitates equitable comparison between base model capabilities and tool-augmented reasoning performance.

## Evaluation Methodology

### Scoring System

The framework implements a binary scoring system (0.0 or 1.0) that balances evaluation rigor with fairness considerations.

### Evaluation Judge Configuration

The evaluation judge utilizes a specialized LLM (Gemini-2.5-Flash) configured with temperature 0 to ensure consistent evaluation. The judge prompt incorporates:

- **Evaluation Rules**: Detailed criteria for determining correctness
- **Scoring Guidelines**: Explicit instructions for score assignment conditions
- **Examples**: Concrete examples of correct and incorrect responses with scoring rationale

This configuration ensures that the judge applies evaluation criteria consistently across all comparisons while maintaining flexibility to assess semantic equivalence and partial correctness.

## Conclusion

The CAIA Ask Evaluator Framework provides a systematic, reproducible methodology for evaluating language models on factual question-answering tasks. Through implementation of a structured evaluation pipeline incorporating LLM-based judging and comprehensive response generation mechanisms, the framework facilitates rigorous comparison of model performance across diverse architectures and providers. The framework's dual evaluation modes—non-tool and tool-enabled—offer complementary perspectives on model capabilities, enabling comprehensive assessment of both intrinsic knowledge and tool-augmented reasoning performance.
