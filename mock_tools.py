#!/usr/bin/env python3
"""Mock tools for anonymized CAIA evaluator"""

import asyncio
from typing import Any, Dict, List, Optional
from langchain_core.tools import tool


# Mock tool implementations with anonymized names
@tool
def crypto_data_tool_1(**kwargs) -> str:
    """Generic crypto data tool 1"""
    return f"Mock crypto data result 1 with args: {kwargs}"


@tool
def crypto_data_tool_2(**kwargs) -> str:
    """Generic crypto data tool 2"""
    return f"Mock crypto data result 2 with args: {kwargs}"


@tool
def crypto_data_tool_3(**kwargs) -> str:
    """Generic crypto data tool 3"""
    return f"Mock crypto data result 3 with args: {kwargs}"


@tool
def market_analysis_tool_1(**kwargs) -> str:
    """Generic market analysis tool 1"""
    return f"Mock market analysis result 1 with args: {kwargs}"


@tool
def market_analysis_tool_2(**kwargs) -> str:
    """Generic market analysis tool 2"""
    return f"Mock market analysis result 2 with args: {kwargs}"


@tool
def blockchain_data_tool_1(**kwargs) -> str:
    """Generic blockchain data tool 1"""
    return f"Mock blockchain data result 1 with args: {kwargs}"


@tool
def blockchain_data_tool_2(**kwargs) -> str:
    """Generic blockchain data tool 2"""
    return f"Mock blockchain data result 2 with args: {kwargs}"


@tool
def blockchain_data_tool_3(**kwargs) -> str:
    """Generic blockchain data tool 3"""
    return f"Mock blockchain data result 3 with args: {kwargs}"


@tool
def social_sentiment_tool(**kwargs) -> str:
    """Generic social sentiment analysis tool"""
    return f"Mock social sentiment result with args: {kwargs}"


@tool
def search_tool(**kwargs) -> str:
    """Generic search tool"""
    return f"Mock search result with args: {kwargs}"


@tool
def web_crawl_tool(**kwargs) -> str:
    """Generic web crawling tool"""
    return f"Mock web crawl result with args: {kwargs}"


@tool
def code_execution_tool(**kwargs) -> str:
    """Generic code execution tool"""
    return f"Mock code execution result with args: {kwargs}"


# Reduced anonymized tool list - only shows general categories
ANONYMIZED_TOOLS = [
    crypto_data_tool_1,
    crypto_data_tool_2,
    crypto_data_tool_3,
    market_analysis_tool_1,
    market_analysis_tool_2,
    blockchain_data_tool_1,
    blockchain_data_tool_2,
    blockchain_data_tool_3,
    social_sentiment_tool,
    search_tool,
    web_crawl_tool,
    code_execution_tool,
]
