"""
LangChain Chatbot - Configured for vLLM Server
Uses TinyLlama model running on vLLM (self-hosted, no API key needed)
"""

import sys
import os
from typing import Dict

# =============================================================================
# VERSION DETECTION AND IMPORTS
# =============================================================================

class VersionAwareImporter:
    """Smart imports for different LangChain versions"""
    
    @staticmethod
    def detect_version():
        try:
            import langchain
            version_str = langchain.__version__
            major, minor = map(int, version_str.split('.')[:2])
            return major, minor, version_str
        except Exception as e:
            print(f"❌ Error detecting LangChain: {e}")
            return None, None, None
    
    @staticmethod
    def import_llm():
        """Import ChatOpenAI (vLLM is compatible with OpenAI API)"""
        try:
            from langchain_openai import ChatOpenAI
            print("✅ Using: langchain_openai.ChatOpenAI (for vLLM)")
            return ChatOpenAI
        except ImportError:
            try:
                from langchain.chat_models import ChatOpenAI
                print("✅ Using: langchain.chat_models.ChatOpenAI (for vLLM)")
                return ChatOpenAI
            except ImportError:
                print("❌ Could not import ChatOpenAI")
                return None
    
    @staticmethod
    def import_chains():
        try:
            from langchain.chains import LLMChain
            return LLMChain
        except ImportError:
            try:
                from langchain_core.chains import LLMChain
                return LLMChain
            except ImportError:
                return None
    
    @staticmethod
    def import_prompts():
        try:
            from langchain.prompts import ChatPromptTemplate
            return ChatPromptTemplate
        except ImportError:
            try:
                from langchain_core.prompts import ChatPromptTemplate
                return ChatPromptTemplate
            except ImportError:
                return None

# Detect and import
print("🔍 Detecting LangChain version and importing modules...\n")

major, minor, version_str = VersionAwareImporter.detect_version()

if major is None:
    print("❌ LangChain not installed!")
    print("Fix with: pip install langchain langchain-openai langchain-core")
    sys.exit(1)

print(f"📦 LangChain version: {version_str}\n")

ChatOpenAI = VersionAwareImporter.import_llm()
LLMChain = VersionAwareImporter.import_chains()
ChatPromptTemplate = VersionAwareImporter.import_prompts()

print()

if not ChatOpenAI:
    print("❌ ChatOpenAI not available")
    sys.exit(1)

# =============================================================================
# CHECKER CLASSES
# =============================================================================

import ast
import json

LANGUAGE_TOOL_AVAILABLE = False
try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
    print("✅ language-tool-python available")
except ImportError:
    print("⚠️  language-tool-python not available (grammar checking will be disabled)")

print()


class GrammarChecker:
    """Task 1: Grammar Checking using LanguageTool"""
    
    def __init__(self):
        if not LANGUAGE_TOOL_AVAILABLE:
            self.available = False
            return
        
        try:
            self.tool = language_tool_python.LanguageTool('en-US')
            self.available = True
        except Exception as e:
            self.available = False
    
    def check(self, text: str) -> Dict:
        """Check grammar"""
        if not self.available:
            return {
                "status": "unavailable",
                "message": "language-tool-python not installed",
                "errors": []
            }
        
        try:
            matches = self.tool.check(text)
            if not matches:
                return {
                    "status": "pass",
                    "message": "No grammar issues found",
                    "errors": []
                }
            
            errors = []
            for match in matches:
                errors.append({
                    "type": match.ruleId,
                    "message": match.message,
                    "suggestion": match.replacements[:3] if match.replacements else [],
                })
            
            return {
                "status": "fail",
                "message": f"Found {len(errors)} grammar issue(s)",
                "errors": errors
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Grammar check failed: {str(e)}",
                "errors": []
            }


class SyntaxChecker:
    """Task 2: Syntax Checking for Python Code"""
    
    def check(self, code: str) -> Dict:
        """Check Python syntax"""
        try:
            ast.parse(code)
            return {
                "status": "pass",
                "message": "Code syntax is valid",
                "errors": [],
                "code_type": "python"
            }
        except SyntaxError as e:
            return {
                "status": "fail",
                "message": "Syntax error found",
                "errors": [{
                    "type": "SyntaxError",
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                }],
                "code_type": "python"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking syntax: {str(e)}",
                "errors": []
            }


class SemanticsChecker:
    """Task 3: Semantic Analysis using vLLM"""
    
    def __init__(self, llm):
        self.llm = llm
        self.use_chains = LLMChain is not None and ChatPromptTemplate is not None
        
        if self.use_chains:
            try:
                semantic_prompt = ChatPromptTemplate.from_template("""
Analyze the semantic meaning and logical consistency of the following text:

Text: {text}

Provide analysis in this format:
1. **Core Meaning**: Brief summary of what the text means
2. **Logical Issues**: Any contradictions or unclear logic
3. **Semantic Clarity**: How well the meaning is conveyed (1-10)
4. **Suggestions**: How to improve semantic clarity

Be concise and practical.
""")
                self.chain = LLMChain(llm=self.llm, prompt=semantic_prompt)
            except Exception as e:
                self.use_chains = False
                self.chain = None
        else:
            self.chain = None
    
    def check(self, text: str) -> Dict:
        """Analyze semantic meaning"""
        try:
            if self.use_chains and self.chain:
                result = self.chain.run(text=text)
            else:
                # Direct invocation
                try:
                    from langchain_core.messages import HumanMessage
                except ImportError:
                    from langchain.schema import HumanMessage
                
                message = HumanMessage(content=f"""Analyze the semantic meaning and logical consistency of the following text:

Text: {text}

Provide analysis in this format:
1. **Core Meaning**: Brief summary of what the text means
2. **Logical Issues**: Any contradictions or unclear logic
3. **Semantic Clarity**: How well the meaning is conveyed (1-10)
4. **Suggestions**: How to improve semantic clarity

Be concise and practical.""")
                response = self.llm.invoke([message])
                result = response.content
            
            return {
                "status": "success",
                "message": "Semantic analysis complete",
                "analysis": result
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in semantic analysis: {str(e)}",
                "analysis": ""
            }


class UniversalChecker:
    """Unified checker that routes to appropriate task"""
    
    def __init__(self, llm):
        self.grammar_checker = GrammarChecker()
        self.syntax_checker = SyntaxChecker()
        self.semantics_checker = SemanticsChecker(llm)
        self.llm = llm
    
    def detect_type(self, text: str) -> str:
        """Detect if text is code or natural language"""
        code_indicators = [
            'def ', 'class ', 'import ', 'return ', 'for ', 'while ',
            'if __name__', '()', '{}', '[]', '=>', 'function ', 'const ',
            'async ', 'await ', 'try:', 'except ', 'if ', 'else:', 'elif '
        ]
        
        text_stripped = text.strip()
        code_count = sum(1 for indicator in code_indicators if indicator in text_stripped)
        
        lines = text_stripped.split('\n')
        indent_lines = sum(1 for line in lines if line and line[0] in (' ', '\t'))
        
        # Try to parse as Python - if it works, it's code
        try:
            ast.parse(text_stripped)
            return "code"
        except (SyntaxError, ValueError):
            pass
        
        # Heuristic detection
        if code_count >= 1 or (indent_lines > len(lines) * 0.3):
            return "code"
        
        return "text"
    
    def check_all(self, text: str) -> Dict:
        """Run all checks"""
        text_type = self.detect_type(text)
        
        results = {
            "input": text,
            "detected_type": text_type,
            "checks": {}
        }
        
        results["checks"]["grammar"] = self.grammar_checker.check(text)
        
        # Always check syntax (helpful for both code and text with code snippets)
        results["checks"]["syntax"] = self.syntax_checker.check(text)
        
        results["checks"]["semantics"] = self.semantics_checker.check(text)
        
        return results


class CheckerChatbot:
    """Main chatbot using vLLM backend"""
    
    def __init__(self, 
                 #vllm_url: str = "https://vllm-xpk.apps.prod.rhoai.rh-aiservices-bu.com/v1",
                 vllm_url: str = "https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1",
                 #model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 model: str = "llama-32-3b-instruct",
                 api_key: str = "not-needed"):
        """
        Initialize chatbot with vLLM server
        
        Args:
            vllm_url: vLLM server endpoint (without /chat/completions)
            model: Model name on vLLM server
            api_key: API key (vLLM usually doesn't require one)
        """
        
        print(f"🚀 Initializing chatbot with vLLM backend...")
        print(f"   Server: {vllm_url}")
        print(f"   Model: {model}")
        
        try:
            # ChatOpenAI can be configured to use any OpenAI-compatible API
            # vLLM provides an OpenAI-compatible endpoint
            self.llm = ChatOpenAI(
                model_name=model,
                openai_api_base=vllm_url,
                openai_api_key=api_key,
                temperature=0.3,
                max_tokens=1024
            )
            print("✅ ChatOpenAI initialized with vLLM backend")
        except Exception as e:
            print(f"❌ Error initializing chatbot: {e}")
            print(f"   Check if vLLM server is running at: {vllm_url}")
            raise
        
        self.universal_checker = UniversalChecker(self.llm)
        self.vllm_url = vllm_url
        self.model = model
    
    def check_text(self, text: str) -> Dict:
        """Check text with all three tasks"""
        return self.universal_checker.check_all(text)
    
    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            "vllm_url": self.vllm_url,
            "model": self.model,
            "temperature": 0.3,
            "max_tokens": 1024
        }


# =============================================================================
# EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LANGCHAIN CHATBOT - vLLM BACKEND (TinyLlama)")
    print("="*70)
    
    # Configuration
    #VLLM_URL = "https://vllm-xpk.apps.prod.rhoai.rh-aiservices-bu.com/v1"
    VLLM_URL = "https://llama-32-3b-instruct-my-first-model.apps.ocp.gp6sl.sandbox2065.opentlc.com/v1"
    MODEL = "llama-32-3b-instruct"
    #MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Initialize
    print("\n🔧 Configuration:")
    print(f"   vLLM URL: {VLLM_URL}")
    print(f"   Model: {MODEL}")
    print()
    
    try:
        chatbot = CheckerChatbot(vllm_url=VLLM_URL, model=MODEL)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\n📝 Troubleshooting:")
        print("   1. Check if vLLM server is running")
        print("   2. Verify URL is correct")
        print("   3. Check network connectivity")
        sys.exit(1)
    
    # Example 1: Grammar
    print("\n" + "-"*70)
    print("EXAMPLE 1: Grammar Checking")
    print("-"*70)
    text1 = "He go to the store yesterday."
    result1 = chatbot.check_text(text1)
    print(f"✍️  Input: {text1}")
    print(f"📊 Grammar status: {result1['checks']['grammar']['status']}")
    if result1['checks']['grammar'].get('errors'):
        for error in result1['checks']['grammar']['errors']:
            print(f"   ❌ {error['message']}")
    
    # Example 2: Syntax
    print("\n" + "-"*70)
    print("EXAMPLE 2: Python Syntax")
    print("-"*70)
    code2 = """
def add(a, b):
    return a + b
"""
    result2 = chatbot.check_text(code2)
    print(f"📋 Type: {result2['detected_type']}")
    print(f"✅ Syntax status: {result2['checks']['syntax']['status']}")
    
    # Example 3: Semantics (using vLLM)
    print("\n" + "-"*70)
    print("EXAMPLE 3: Semantic Analysis (using vLLM)")
    print("-"*70)
    text3 = "The algorithm is simple but complex."
    result3 = chatbot.check_text(text3)
    print(f"✍️  Input: {text3}")
    print(f"📊 Status: {result3['checks']['semantics']['status']}")
    if result3['checks']['semantics']['status'] == 'success':
        print(f"📝 Analysis:")
        print(result3['checks']['semantics']['analysis'][:300] + "...")
    else:
        print(f"   Error: {result3['checks']['semantics']['message']}")
    
    print("\n" + "="*70)
    print("✨ Chatbot ready!")
    print("="*70)
    print("\nUsage:")
    print("  from chatbot_vllm import CheckerChatbot")
    print("  chatbot = CheckerChatbot(")
    print(f'      vllm_url="{VLLM_URL}",')
    print(f'      model="{MODEL}"')
    print("  )")
    print("  result = chatbot.check_text('your text')")
    print("\n")