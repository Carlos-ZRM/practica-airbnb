"""
Grammar Chatbot Utility
Provides easy-to-use functions and classes for grammar correction using LangGraph
"""

from langgraph.graph import END, StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict, Dict, Any, List, Optional
import json


class AgentState(TypedDict):
    """Define the state structure for the workflow"""
    query: str
    query_feedback: str
    grammar_query: str
    correction_confidence: float  # Confidence score 0.0-1.0
    quality_score: float  # Quality rating 0.0-1.0


class GrammarBot:
    def __init__(self, llm):
        self.llm = llm
        self.workflow_app = None

    def fix_syntax(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fixes the morphological issues in the user query to improve quality of strings

        Args:
            state (Dict): The current state of the workflow, containing the user query.

        Returns:
            Dict: The updated state with the fixed query.
        """
        print("---------Fixing Syntax Issues---------")
        
        system_message = '''Tu eres un experto en procesamiento de lenguaje natural especializado 
        en mejorar la calidad de las consultas de los usuarios. Tu tarea es corregir los errores morfológicos
        presentes en la consulta del usuario para mejorar su calidad y claridad.
        
        Guias:
        - Debes identificar los elementos sintácticos y revisar que estén unidos incorrectamente.
        - Debes corregir las palabras que estén mal formadas debido a errores morfológicos.
        - Debes mantener el significado original de la consulta mientras corriges los errores morfológicos
        - Debes asegurarte de que la consulta corregida sea gramaticalmente correcta y fácil de entender.
        - Debes proporcionar la consulta corregida sin agregar explicaciones adicionales.
        - Debes responder únicamente con la consulta corregida, sin incluir ningún otro texto o formato.
        '''

        fix_grammar_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Corrige esta sentencia: {query}")
        ])

        chain = fix_grammar_prompt | self.llm | StrOutputParser()
        
        result_query = chain.invoke({"query": state['query']})
        
        print("grammar_query:", result_query)
        state["grammar_query"] = result_query
        
        return state

    def classify_text(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify and rate the correction quality
        Provides confidence score and quality metrics
        
        Args:
            state (Dict): The current state with original and corrected queries
            
        Returns:
            Dict: Updated state with classification scores
        """
        import json  # Import at method level
        
        print("---------Classifying Text Quality---------")
        
        original_text = state.get('query', '')
        corrected_text = state.get('grammar_query', '')
        
        if not corrected_text:
            # If no correction was made, return default scores
            state['correction_confidence'] = 0.0
            state['quality_score'] = 0.0
            return state
        
        system_message = '''Tu eres un experto en evaluación de la calidad del lenguaje natural.
Tu tarea es evaluar la calidad de una corrección gramatical.

Proporciona dos métricas (ambas entre 0.0 y 1.0):

1. correction_confidence: Confianza en que la corrección es correcta (0.0=sin confianza, 1.0=muy confiante)
2. quality_score: Puntuación de calidad del texto corregido (0.0=muy pobre, 1.0=excelente)

Considera:
- Correctitud gramatical
- Claridad del texto
- Naturalidad del lenguaje
- Preservación del significado original
- Coherencia y fluidez

Responde ÚNICAMENTE con un JSON válido en este formato, sin explicaciones adicionales:
{{"correction_confidence": 0.85, "quality_score": 0.90}}
'''
        
        classify_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("user", "Original: {original}\nCorregido: {corrected}\n\nProporciona únicamente el JSON con los dos scores (correction_confidence y quality_score entre 0.0 y 1.0).")
        ])
        
        chain = classify_prompt | self.llm | StrOutputParser()
        
        try:
            result_str = chain.invoke({
                "original": original_text,
                "corrected": corrected_text
            })
            
            print("Classification result:", result_str)
            
            # Clean the response (remove markdown formatting if present)
            result_str = result_str.strip()
            if result_str.startswith('```'):
                result_str = result_str.split('```')[1]
                if result_str.startswith('json'):
                    result_str = result_str[4:]
            
            result_str = result_str.strip()
            
            result_json = json.loads(result_str)
            
            # Extract scores with validation
            confidence = float(result_json.get('correction_confidence', 0.5))
            quality = float(result_json.get('quality_score', 0.5))
            
            # Ensure scores are between 0.0 and 1.0
            confidence = max(0.0, min(1.0, confidence))
            quality = max(0.0, min(1.0, quality))
            
            state['correction_confidence'] = confidence
            state['quality_score'] = quality
            
            print(f"Confidence: {confidence:.2f}, Quality: {quality:.2f}")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {result_str}")
            # Fallback to default scores
            state['correction_confidence'] = 0.5
            state['quality_score'] = 0.5
        except Exception as e:
            print(f"Error in classification: {e}")
            # Fallback to default scores
            state['correction_confidence'] = 0.5
            state['quality_score'] = 0.5
        
        return state

    def create_workflow(self):
        """Create and compile the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("fix_syntax", self.fix_syntax)
        workflow.add_node("classify_text", self.classify_text)
        
        # Add edges
        workflow.set_entry_point("fix_syntax")
        workflow.add_edge("fix_syntax", "classify_text")
        workflow.add_edge("classify_text", END)
        
        # Compile the workflow
        self.workflow_app = workflow.compile()
        
        return self.workflow_app

    def correct_text(self, text: str, feedback: str = "") -> str:
        """
        Simple method to correct a single text string
        
        Args:
            text (str): The text to correct
            feedback (str): Optional feedback about the correction
            
        Returns:
            str: The corrected text
        """
        if self.workflow_app is None:
            self.create_workflow()
        
        initial_state = {
            "query": text,
            "query_feedback": feedback,
            "grammar_query": "",
            "correction_confidence": 0.0,
            "quality_score": 0.0
        }
        
        result = self.workflow_app.invoke(initial_state)
        print("MY Correction Result:", result)
        return result
        #return result["grammar_query"]

    def correct_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Correct multiple texts and return results with scores
        
        Args:
            texts (List[str]): List of texts to correct
            
        Returns:
            List[Dict]: List of dicts with 'original', 'corrected', 'confidence', 'quality' keys
        """
        if self.workflow_app is None:
            self.create_workflow()
        
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n[{i}/{len(texts)}] Processing: {text[:50]}...")
            corrected_result = self.workflow_app.invoke({
                "query": text,
                "query_feedback": "",
                "grammar_query": "",
                "correction_confidence": 0.0,
                "quality_score": 0.0
            })
            
            results.append({
                "original": text,
                "corrected": corrected_result["grammar_query"],
                "confidence": corrected_result["correction_confidence"],
                "quality": corrected_result["quality_score"]
            })
        
        #print("Results:", json.dumps(results, indent=2, ensure_ascii=False))
        return results

    def interactive_mode(self):
        """
        Launch an interactive terminal session for real-time grammar correction
        """
        if self.workflow_app is None:
            self.create_workflow()
        
        print("\n" + "="*70)
        print("Grammar Chatbot - Interactive Mode")
        print("="*70)
        print("Type your text to correct (type 'exit' or 'quit' to exit)")
        print("-"*70 + "\n")
        
        while True:
            user_input = input("Enter text to correct: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                print("Please enter some text.\n")
                continue
            
            corrected = self.correct_text(user_input)
            print(f"\nOriginal:  {user_input}")
            print(f"Corrected: {corrected}\n")
            print("-"*70 + "\n")

    def print_graph(self, format_type: str = "ascii"):
        """
        Print the LangGraph workflow structure
        
        Args:
            format_type (str): Type of visualization - 'ascii', 'mermaid', or 'details'
        """
        if self.workflow_app is None:
            self.create_workflow()
        
        print("\n" + "="*70)
        print("Grammar Chatbot - Workflow Graph")
        print("="*70)
        
        if format_type == "ascii":
            self._print_ascii_graph()
        elif format_type == "mermaid":
            self._print_mermaid_graph()
        elif format_type == "details":
            self._print_detailed_graph()
        else:
            print(f"Unknown format: {format_type}")
            print("Supported formats: 'ascii', 'mermaid', 'details'")
        
        print("="*70 + "\n")

    def _print_ascii_graph(self):
        """Print ASCII representation of the graph"""
        try:
            # Try using the built-in ASCII visualization
            graph = self.workflow_app.get_graph()
            ascii_art = graph.draw_ascii()
            print(ascii_art)
        except AttributeError:
            # Fallback ASCII visualization
            print("\nWorkflow Structure (ASCII):")
            print("""
    START
      │
      ▼
┌─────────────────────┐
│   fix_syntax        │
│   (Grammar Node)    │
└─────────────────────┘
      │
      ▼
┌──────────────────────┐
│  classify_text       │
│  (Rating Node)       │
└──────────────────────┘
      │
      ▼
     END
            """)

    def _print_mermaid_graph(self):
        """Print Mermaid diagram representation of the graph"""
        try:
            graph = self.workflow_app.get_graph()
            mermaid_diagram = graph.draw_mermaid()
            print("\nWorkflow Structure (Mermaid):")
            print(mermaid_diagram)
        except AttributeError:
            # Fallback Mermaid visualization
            print("\nWorkflow Structure (Mermaid):")
            print("""
graph TD
    START([START])
    FIX[fix_syntax<br/>Grammar Correction Node]
    CLASSIFY[classify_text<br/>Quality Rating Node]
    END([END])
    
    START --> FIX
    FIX --> CLASSIFY
    CLASSIFY --> END
    
    style FIX fill:#4CAF50,stroke:#2E7D32,color:#fff
    style CLASSIFY fill:#FF9800,stroke:#E65100,color:#fff
    style START fill:#2196F3,stroke:#1565C0,color:#fff
    style END fill:#F44336,stroke:#C62828,color:#fff
            """)

    def _print_detailed_graph(self):
        """Print detailed information about the graph"""
        print("\nWorkflow Detailed Information:")
        print("-" * 70)
        
        try:
            graph = self.workflow_app.get_graph()
            
            # Graph metadata
            print("\n📊 GRAPH METADATA:")
            print(f"  Graph Type: StateGraph")
            print(f"  State Type: AgentState")
            print(f"  Nodes: 2 (Grammar + Classification)")
            
            # Nodes
            print("\n🔧 NODES:")
            print(f"  • fix_syntax")
            print(f"    ├─ Type: Grammar Correction Node")
            print(f"    ├─ Function: Fixes morphological issues in text")
            print(f"    ├─ Output: grammar_query (corrected text)")
            print(f"    └─ Input/Output: Dict[str, Any]")
            print(f"")
            print(f"  • classify_text")
            print(f"    ├─ Type: Quality Rating Node")
            print(f"    ├─ Function: Evaluates correction quality")
            print(f"    ├─ Output: correction_confidence, quality_score")
            print(f"    └─ Input/Output: Dict[str, Any]")
            
            # Edges
            print("\n🔗 EDGES (Connections):")
            print(f"  START → fix_syntax → classify_text → END")
            
            # State fields
            print("\n📋 STATE SCHEMA (AgentState):")
            print(f"  • query: str")
            print(f"    └─ The original text to correct")
            print(f"  • query_feedback: str")
            print(f"    └─ Optional feedback about the correction")
            print(f"  • grammar_query: str")
            print(f"    └─ The corrected text result")
            print(f"  • correction_confidence: float")
            print(f"    └─ How confident in the correction (0.0-1.0)")
            print(f"  • quality_score: float")
            print(f"    └─ Overall quality rating (0.0-1.0)")
            
            # Flow description
            print("\n📈 WORKFLOW FLOW:")
            print(f"  1. START: Initialize AgentState with:")
            print(f"     - query, query_feedback, grammar_query, scores (0.0)")
            print(f"  2. fix_syntax: Grammar correction node")
            print(f"     - Corrects text using LLM")
            print(f"     - Updates: grammar_query field")
            print(f"  3. classify_text: Quality rating node")
            print(f"     - Evaluates correction quality")
            print(f"     - Updates: correction_confidence, quality_score")
            print(f"  4. END: Return final state with all fields")
            
        except Exception as e:
            print(f"Error printing detailed graph: {e}")

    def print_graph_summary(self):
        """Print a quick summary of the graph"""
        print("\n" + "="*70)
        print("Grammar Chatbot - Quick Summary")
        print("="*70)
        
        print("\n✓ Workflow: Two-node Grammar Correction Pipeline")
        print("✓ Nodes:")
        print("  1. fix_syntax - Grammar correction (LLM-based)")
        print("  2. classify_text - Quality assessment (LLM-based)")
        print("✓ State: AgentState with 5 fields")
        print("  - query (text), query_feedback, grammar_query")
        print("  - correction_confidence (0.0-1.0)")
        print("  - quality_score (0.0-1.0)")
        print("✓ Flow: START → fix_syntax → classify_text → END")
        print("✓ Status: Compiled and ready to use")
        
        print("\n" + "="*70 + "\n")


# Convenience function for one-off corrections
def correct_grammar(text: str, llm) -> str:
    """
    Quick function to correct grammar without creating a bot instance
    
    Args:
        text (str): Text to correct
        llm: LangChain LLM instance
        
    Returns:
        str: Corrected text
    """
    bot = GrammarBot(llm)
    bot.create_workflow()
    return bot.correct_text(text)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    
    # Initialize the LLM with Red Hat AI Services vLLM endpoint
    llm = ChatOpenAI(
        model_name="llama-32-3b-instruct",
        openai_api_base="https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
        openai_api_key="your-api-key",  # Replace with your actual API key
        temperature=0.0,
    )
    
    # ========================================================================
    # EXAMPLE 0: View Bot Graph Structure
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 0: Bot Graph Visualization")
    print("="*70)
    
    bot = GrammarBot(llm)
    bot.create_workflow()
    
    # Print graph in different formats
    print("\n--- ASCII Format ---")
    bot.print_graph(format_type="ascii")
     
    print("\n--- Mermaid Format ---")
    bot.print_graph(format_type="mermaid")
    
    print("\n--- Detailed Format ---")
    bot.print_graph(format_type="details")
    
    print("\n--- Quick Summary ---")
    bot.print_graph_summary()
    
    # ========================================================================
    # EXAMPLE 1: Single text correction
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Text Correction")
    print("="*70)
    
    text = "El usurio quere corregir el sntaxys de la consulta"
    corrected = bot.correct_text(text)
    print(f"Original:  {text}")
    print(f"Corrected: {corrected}")
    
    # ========================================================================
    # EXAMPLE 2: Batch correction
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Text Correction")
    print("="*70)
    
    texts_to_correct = [
        "El usurio quere corregir el sntaxys",
        "Yo tengo un eror en mi tarea",
        "La gramatica es importent para comunkcar bien"
    ]
    
    batch_results = bot.correct_batch(texts_to_correct)
    
    print("\nBatch Results:")
    print(json.dumps(batch_results, indent=2, ensure_ascii=False))
    
    # ========================================================================
    # EXAMPLE 3: Quick correction using convenience function
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 3: Quick Correction (One-liner)")
    print("="*70)
    
    result = correct_grammar("El gato estan durmiendo en la cama", llm)
    print(f"Corrected: {result}")
    
    # ========================================================================
    # EXAMPLE 4: Interactive mode (uncomment to use)
    # ========================================================================
    # print("\n" + "="*70)
    # print("EXAMPLE 4: Interactive Mode")
    # print("="*70)
    # bot.interactive_mode()