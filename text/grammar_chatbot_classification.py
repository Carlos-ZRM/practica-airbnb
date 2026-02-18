"""
Grammar Chatbot - Classification & Rating Node
==============================================

Demonstrates the new classify_text node that provides:
  - correction_confidence: How confident the correction is (0.0-1.0)
  - quality_score: Overall quality rating (0.0-1.0)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from grammar_chatbot_utils import GrammarBot
except ImportError:
    # If still not found, provide helpful error message
    print("ERROR: Could not import grammar_chatbot_utils")
    print("Make sure grammar_chatbot_utils.py is in the same directory")
    print("Current directory:", Path.cwd())
    print("Script directory:", Path(__file__).parent)
    sys.exit(1)

from langchain_openai import ChatOpenAI
import json
from typing import Dict, List


def get_llm():
    """Get configured LLM instance"""
    return ChatOpenAI(
        model_name="llama-32-3b-instruct",
        openai_api_base="https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
        openai_api_key="your-api-key",
        temperature=0.0,
    )


# ============================================================================
# EXAMPLE 1: Simple correction with classification scores
# ============================================================================

def example_simple_classification():
    """Correct text and get quality scores"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Correction with Scores")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    text = "El usurio quere corregir el sntaxys de la consulta"
    
    # Run the workflow
    result = bot.workflow_app.invoke({
        "query": text,
        "query_feedback": "",
        "grammar_query": "",
        "correction_confidence": 0.0,
        "quality_score": 0.0
    })
    
    print(f"\nOriginal:  {result['query']}")
    print(f"Corrected: {result['grammar_query']}")
    print(f"\n📊 SCORES:")
    print(f"  Confidence: {result['correction_confidence']:.2f} ({result['correction_confidence']*100:.0f}%)")
    print(f"  Quality:    {result['quality_score']:.2f} ({result['quality_score']*100:.0f}%)")
    print(f"\nRating: {'⭐' * int(result['quality_score'] * 5)}")


# ============================================================================
# EXAMPLE 2: Batch processing with scores
# ============================================================================

def example_batch_with_scores():
    """Process multiple texts with quality scores"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Processing with Scores")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    texts = [
        "El usurio quere corregir el sntaxys",
        "Yo tengo un eror en mi tarea",
        "La gramatica es importent"
    ]
    
    results = bot.correct_batch(texts)
    
    # Display results in table format
    print("\n" + "-"*70)
    print(f"{'Text':<30} {'Confidence':<15} {'Quality':<15}")
    print("-"*70)
    
    for result in results:
        original = result['original'][:25] + "..." if len(result['original']) > 25 else result['original']
        conf = f"{result['confidence']:.2f}"
        qual = f"{result['quality']:.2f}"
        print(f"{original:<30} {conf:<15} {qual:<15}")
    
    print("-"*70)
    
    # Show detailed results
    print("\nDETAILED RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Original: {result['original']}")
        print(f"    Corrected: {result['corrected']}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Quality: {result['quality']:.2f}")


# ============================================================================
# EXAMPLE 3: Quality filtering
# ============================================================================

def example_quality_filtering():
    """Filter corrections by quality threshold"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Quality Filtering")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    texts = [
        "El usurio quere",
        "Tengo un eror",
        "Gramatica importent",
        "Hola mundo"
    ]
    
    quality_threshold = 0.7  # Only accept corrections with quality >= 0.7
    
    results = bot.correct_batch(texts)
    
    print(f"\nQuality Threshold: {quality_threshold:.2f}")
    print("-"*70)
    
    high_quality = [r for r in results if r['quality'] >= quality_threshold]
    low_quality = [r for r in results if r['quality'] < quality_threshold]
    
    print(f"\n✓ HIGH QUALITY ({len(high_quality)} results):")
    for result in high_quality:
        print(f"  • {result['original']}")
        print(f"    → {result['corrected']}")
        print(f"    Quality: {result['quality']:.2f}")
    
    print(f"\n✗ LOW QUALITY ({len(low_quality)} results) - NEEDS REVIEW:")
    for result in low_quality:
        print(f"  • {result['original']}")
        print(f"    → {result['corrected']}")
        print(f"    Quality: {result['quality']:.2f}")


# ============================================================================
# EXAMPLE 4: Confidence-based confidence intervals
# ============================================================================

def example_confidence_analysis():
    """Analyze correction confidence levels"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Confidence Analysis")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    texts = [
        "El usurio quere corregir",
        "Tengo un eror",
        "Hola mundo",
        "Gramatica es importent"
    ]
    
    results = bot.correct_batch(texts)
    
    # Categorize by confidence
    very_confident = [r for r in results if r['confidence'] >= 0.8]
    confident = [r for r in results if 0.6 <= r['confidence'] < 0.8]
    uncertain = [r for r in results if r['confidence'] < 0.6]
    
    print(f"\nConfidence Distribution:")
    print(f"  Very Confident (≥0.8): {len(very_confident)}")
    print(f"  Confident (0.6-0.8): {len(confident)}")
    print(f"  Uncertain (<0.6): {len(uncertain)}")
    
    print(f"\n🔒 VERY CONFIDENT CORRECTIONS:")
    for result in very_confident:
        print(f"  {result['original']} → {result['corrected']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    print(f"\n⚠️  UNCERTAIN CORRECTIONS (REVIEW RECOMMENDED):")
    for result in uncertain:
        print(f"  {result['original']} → {result['corrected']}")
        print(f"  Confidence: {result['confidence']:.2f}")


# ============================================================================
# EXAMPLE 5: Score visualization
# ============================================================================

def example_score_visualization():
    """Visualize scores as bars"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Score Visualization")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    text = "El usurio quere corregir el sntaxys"
    
    result = bot.workflow_app.invoke({
        "query": text,
        "query_feedback": "",
        "grammar_query": "",
        "correction_confidence": 0.0,
        "quality_score": 0.0
    })
    
    confidence = result['correction_confidence']
    quality = result['quality_score']
    
    def score_bar(score: float, width: int = 30) -> str:
        """Create a visual bar for score"""
        filled = int(score * width)
        empty = width - filled
        return "█" * filled + "░" * empty
    
    print(f"\nOriginal: {result['query']}")
    print(f"Corrected: {result['grammar_query']}")
    
    print(f"\n📊 SCORES VISUALIZATION:")
    print(f"\nConfidence:")
    print(f"  [{score_bar(confidence)}] {confidence:.1%}")
    
    print(f"\nQuality:")
    print(f"  [{score_bar(quality)}] {quality:.1%}")
    
    # Overall rating
    avg_score = (confidence + quality) / 2
    print(f"\nOverall Rating:")
    print(f"  [{score_bar(avg_score)}] {avg_score:.1%}")
    
    # Star rating
    stars = int(avg_score * 5)
    print(f"  Stars: {'⭐' * stars}{'☆' * (5 - stars)}")


# ============================================================================
# EXAMPLE 6: Export results to JSON
# ============================================================================

def example_export_results():
    """Export correction results with scores to JSON"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Export Results to JSON")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    texts = [
        "El usurio quere corregir",
        "Tengo un eror",
        "Gramatica importent"
    ]
    
    results = bot.correct_batch(texts)
    
    # Create export structure
    export_data = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "total_corrections": len(results),
        "average_confidence": sum(r['confidence'] for r in results) / len(results),
        "average_quality": sum(r['quality'] for r in results) / len(results),
        "corrections": results
    }
    
    # Display JSON
    print("\nJSON Export:")
    print(json.dumps(export_data, indent=2, ensure_ascii=False))


# ============================================================================
# EXAMPLE 7: Quality metrics report
# ============================================================================

def example_quality_report():
    """Generate comprehensive quality metrics report"""
    print("\n" + "="*70)
    print("EXAMPLE 7: Quality Metrics Report")
    print("="*70)
    
    llm = get_llm()
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    texts = [
        "El usurio quere corregir el sntaxys",
        "Yo tengo un eror en mi tarea",
        "La gramatica es importent para comunicar",
        "El gato estan durmiendo",
        "Esto es un texto sin errores"
    ]
    
    results = bot.correct_batch(texts)
    
    # Calculate metrics
    confidences = [r['confidence'] for r in results]
    qualities = [r['quality'] for r in results]
    
    avg_conf = sum(confidences) / len(confidences)
    avg_qual = sum(qualities) / len(qualities)
    
    # Find min/max
    min_conf = min(confidences)
    max_conf = max(confidences)
    min_qual = min(qualities)
    max_qual = max(qualities)
    
    print("\n" + "─"*70)
    print("QUALITY METRICS SUMMARY")
    print("─"*70)
    
    print(f"\nSamples Processed: {len(results)}")
    
    print(f"\n📊 CONFIDENCE METRICS:")
    print(f"  Average:  {avg_conf:.2f}")
    print(f"  Min:      {min_conf:.2f}")
    print(f"  Max:      {max_conf:.2f}")
    print(f"  Range:    {max_conf - min_conf:.2f}")
    
    print(f"\n📈 QUALITY METRICS:")
    print(f"  Average:  {avg_qual:.2f}")
    print(f"  Min:      {min_qual:.2f}")
    print(f"  Max:      {max_qual:.2f}")
    print(f"  Range:    {max_qual - min_qual:.2f}")
    
    print(f"\n🎯 OVERALL SCORE:")
    overall = (avg_conf + avg_qual) / 2
    print(f"  {overall:.2f} ({'⭐' * int(overall * 5)})")
    
    print("\n" + "─"*70)


# ============================================================================
# MAIN: Run Examples
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        
        examples = {
            "1": example_simple_classification,
            "2": example_batch_with_scores,
            "3": example_quality_filtering,
            "4": example_confidence_analysis,
            "5": example_score_visualization,
            "6": example_export_results,
            "7": example_quality_report
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found")
    else:
        print("\n" + "="*70)
        print("GRAMMAR CHATBOT - CLASSIFICATION EXAMPLES")
        print("="*70)
        print("\nUsage:")
        print("  python grammar_chatbot_classification.py [1-7]")
        print("\nExamples:")
        print("  1: Simple correction with scores")
        print("  2: Batch processing with scores")
        print("  3: Quality filtering")
        print("  4: Confidence analysis")
        print("  5: Score visualization")
        print("  6: Export to JSON")
        print("  7: Quality metrics report")
        print("\nRun all examples:")
        print("  for i in {1..7}; do python grammar_chatbot_classification.py $i; done")
        print("="*70)