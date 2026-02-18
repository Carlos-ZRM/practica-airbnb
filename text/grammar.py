#!/usr/bin/env python3
"""
Grammar Chatbot - Quick Start Script
===================================

This script demonstrates the complete workflow with proper error handling
and import path management. Run this first to test your setup!
"""

import sys
import os
from pathlib import Path

# ============================================================================
# FIX IMPORTS
# ============================================================================

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("=" * 70)
print("GRAMMAR CHATBOT - QUICK START")
print("=" * 70)
print(f"\nScript directory: {current_dir}")
print(f"Python version: {sys.version.split()[0]}")

# ============================================================================
# CHECK IMPORTS
# ============================================================================

print("\n[1] Checking imports...")
print("-" * 70)

try:
    from langchain_openai import ChatOpenAI
    print("✓ langchain-openai imported")
except ImportError as e:
    print(f"✗ langchain-openai not found: {e}")
    print("\nFix: pip install langchain-openai")
    sys.exit(1)

try:
    from langgraph.graph import StateGraph, END, START
    print("✓ langgraph imported")
except ImportError as e:
    print(f"✗ langgraph not found: {e}")
    print("\nFix: pip install langgraph")
    sys.exit(1)

try:
    from grammar_chatbot_utils import GrammarBot
    print("✓ grammar_chatbot_utils imported")
except ImportError as e:
    print(f"✗ grammar_chatbot_utils not found: {e}")
    print("\nFix: Make sure grammar_chatbot_utils.py is in:")
    print(f"      {current_dir}")
    sys.exit(1)

# ============================================================================
# CHECK FILES
# ============================================================================

print("\n[2] Checking required files...")
print("-" * 70)

required_files = [
    "grammar_chatbot_utils.py",
    "grammar_chatbot_classification.py",
    "grammar_chatbot_graph.py"
]

for filename in required_files:
    filepath = current_dir / filename
    if filepath.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename} NOT FOUND")
        print(f"   Looking in: {filepath}")

# ============================================================================
# INITIALIZE BOT
# ============================================================================

print("\n[3] Initializing Grammar Chatbot...")
print("-" * 70)

# Configure your API credentials here
LLM_CONFIG = {
    "model_name": "llama-32-3b-instruct",
    "openai_api_base": "https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
    "openai_api_key": "your-api-key",  # ⭐ UPDATE THIS
    "temperature": 0.0,
}

# Check if API key is set
if LLM_CONFIG["openai_api_key"] == "your-api-key":
    print("\n⚠️  WARNING: API key not configured!")
    print("\nFix: Edit this script and replace:")
    print('   "openai_api_key": "your-api-key"')
    print("with your actual API key from your vLLM endpoint")
    print("\nOr set environment variable:")
    print('   export OPENAI_API_KEY="your-key"')
    response = input("\nContinue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

try:
    llm = ChatOpenAI(**LLM_CONFIG)
    print("✓ LLM initialized")
    
    bot = GrammarBot(llm)
    print("✓ GrammarBot created")
    
    bot.create_workflow()
    print("✓ Workflow compiled")
    
except Exception as e:
    print(f"✗ Error initializing bot: {e}")
    print("\nTroubleshooting:")
    print("1. Check API key is correct")
    print("2. Check endpoint URL is reachable")
    print("3. Run: python test_vllm_endpoint.py")
    sys.exit(1)

# ============================================================================
# TEST CORRECTION
# ============================================================================

print("\n[4] Testing grammar correction...")
print("-" * 70)

test_text = "El usurio quere corregir el sntaxys"
print(f"\nInput text: {test_text}")

try:
    result = bot.workflow_app.invoke({
        "query": test_text,
        "query_feedback": "",
        "grammar_query": "",
        "correction_confidence": 0.0,
        "quality_score": 0.0
    })
    
    print(f"\n✓ Correction successful!")
    print(f"\nResults:")
    print(f"  Original:   {result['query']}")
    print(f"  Corrected:  {result['grammar_query']}")
    print(f"  Confidence: {result['correction_confidence']:.2f}")
    print(f"  Quality:    {result['quality_score']:.2f}")
    
except Exception as e:
    print(f"\n✗ Error during correction: {e}")
    print("\nTroubleshooting:")
    print("1. Check LLM endpoint is running")
    print("2. Check model name is correct")
    print("3. Check API key has access to model")
    import traceback
    print("\nFull error:")
    print(traceback.format_exc())
    sys.exit(1)

# ============================================================================
# SHOW WORKFLOW GRAPH
# ============================================================================

print("\n[5] Workflow structure...")
print("-" * 70)

bot.print_graph(format_type="ascii")

# ============================================================================
# SUCCESS
# ============================================================================

print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED!")
print("=" * 70)

print("\nYou're ready to use the Grammar Chatbot!")
print("\nNext steps:")
print("1. Edit this script to add your API key")
print("2. Try: python grammar_chatbot_classification.py 1")
print("3. Try: python grammar_chatbot_graph.py")
print("4. See GRAMMAR_CHATBOT_GUIDE.py for more examples")

print("\nUseful commands:")
print("  python test_vllm_endpoint.py          # Verify endpoint")
print("  python grammar_chatbot_snippets.py    # Run snippets")
print("  python grammar_chatbot_examples.py    # See real examples")
print("  python SETUP_AND_TROUBLESHOOTING.py   # Full setup guide")

print("\n" + "=" * 70 + "\n")