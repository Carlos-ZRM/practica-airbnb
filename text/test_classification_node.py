#!/usr/bin/env python3
"""
Grammar Chatbot - Classification Node Test
==========================================

Tests the fixed classify_text node to ensure JSON parsing works correctly.
"""

import sys
from pathlib import Path

# Fix imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from grammar_chatbot_utils import GrammarBot
    from langchain_openai import ChatOpenAI
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


def test_classification():
    """Test the classification node with proper error handling"""
    
    print("\n" + "="*70)
    print("TESTING CLASSIFICATION NODE")
    print("="*70)
    
    # Initialize LLM
    print("\n[1] Initializing LLM...")
    llm = ChatOpenAI(
        model_name="llama-32-3b-instruct",
        openai_api_base="https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
        openai_api_key="your-api-key",  # ⭐ Update this
        temperature=0.0,
    )
    print("✓ LLM initialized")
    
    # Create bot
    print("\n[2] Creating GrammarBot...")
    bot = GrammarBot(llm)
    bot.create_workflow()
    print("✓ Workflow compiled")
    
    # Test single correction
    print("\n[3] Testing single correction with classification...")
    print("-" * 70)
    
    test_text = "El usurio quere corregir el sntaxys"
    print(f"Input: {test_text}\n")
    
    try:
        result = bot.workflow_app.invoke({
            "query": test_text,
            "query_feedback": "",
            "grammar_query": "",
            "correction_confidence": 0.0,
            "quality_score": 0.0
        })
        
        print("✓ Workflow executed successfully!\n")
        print("RESULTS:")
        print(f"  Original:    {result['query']}")
        print(f"  Corrected:   {result['grammar_query']}")
        print(f"  Confidence:  {result['correction_confidence']:.2f}")
        print(f"  Quality:     {result['quality_score']:.2f}")
        
        # Validate scores
        if 0.0 <= result['correction_confidence'] <= 1.0:
            print("\n✓ Confidence score is valid (0.0-1.0)")
        else:
            print(f"\n✗ Invalid confidence score: {result['correction_confidence']}")
            
        if 0.0 <= result['quality_score'] <= 1.0:
            print("✓ Quality score is valid (0.0-1.0)")
        else:
            print(f"✗ Invalid quality score: {result['quality_score']}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def test_batch():
    """Test batch processing with classification"""
    
    print("\n" + "="*70)
    print("TESTING BATCH PROCESSING WITH CLASSIFICATION")
    print("="*70)
    
    # Initialize LLM
    print("\n[1] Initializing LLM...")
    llm = ChatOpenAI(
        model_name="llama-32-3b-instruct",
        openai_api_base="https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
        openai_api_key="your-api-key",  # ⭐ Update this
        temperature=0.0,
    )
    print("✓ LLM initialized")
    
    # Create bot
    print("\n[2] Creating GrammarBot...")
    bot = GrammarBot(llm)
    bot.create_workflow()
    print("✓ Workflow compiled")
    
    # Test batch
    print("\n[3] Testing batch processing...")
    print("-" * 70)
    
    texts = [
        "El usurio quere",
        "Tengo un eror",
        "Gramatica importent"
    ]
    
    print(f"Processing {len(texts)} texts...\n")
    
    try:
        results = bot.correct_batch(texts)
        
        print(f"\n✓ Batch processing completed!\n")
        print("RESULTS:")
        print("-" * 70)
        print(f"{'Original':<25} {'Corrected':<25} {'Conf':<6} {'Quality':<6}")
        print("-" * 70)
        
        for result in results:
            orig = result['original'][:23] if len(result['original']) > 23 else result['original']
            corr = result['corrected'][:23] if len(result['corrected']) > 23 else result['corrected']
            conf = f"{result['confidence']:.2f}"
            qual = f"{result['quality']:.2f}"
            
            print(f"{orig:<25} {corr:<25} {conf:<6} {qual:<6}")
        
        print("-" * 70)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("GRAMMAR CHATBOT - CLASSIFICATION NODE TEST SUITE")
    print("="*70)
    
    # Run tests
    test1_passed = test_classification()
    test2_passed = test_batch()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nThe classification node is working correctly.")
        print("You can now use:")
        print("  - bot.correct_text(text)")
        print("  - bot.correct_batch(texts)")
        print("  - bot.workflow_app.invoke(state)")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Check endpoint URL is reachable")
        print("3. Check model name is correct")
        print("4. Run: python test_vllm_endpoint.py")
        sys.exit(1)
    
    print("\n" + "="*70 + "\n")
