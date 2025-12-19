import os
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache

# 1. Enable Caching (One line replaces your entire _cache logic)
set_llm_cache(InMemoryCache())

class LangChainClient:
    """
    Unified Client using LangChain.
    No need to manually handle retries, API errors, or standardizing responses.
    """
    
    PROVIDER_MAP = {
        "google": ChatGoogleGenerativeAI,
        "openai": ChatOpenAI,
        "anthropic": ChatAnthropic
    }

    def __init__(self, provider: str = "google", model_name: str = "gemini-2.5-pro"):
        self.provider = provider
        self.model_name = model_name
        
        # Factory pattern to pick the right class
        model_class = self.PROVIDER_MAP.get(provider)
        if not model_class:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize the model
        # max_retries is built-in! No need for 'tenacity' decorators.
        self.llm: BaseChatModel = model_class(
            model=model_name,
            max_retries=3,
            api_key=self._get_api_key(provider)
        )

    def _get_api_key(self, provider: str) -> str:
        """Helper to fetch keys from env"""
        keys = {
            "google": os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")
        }
        return keys.get(provider)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Standard text generation"""
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        # Invoke the model
        response = self.llm.invoke(messages)
        
        # LangChain standardizes the content for us
        return response.content

    def generate_json(self, prompt: str, schema: Dict[str, Any]) -> Dict:
        """
        Structured Output (The Killer Feature)
        LangChain handles the "json_schema" vs "function_calling" logic for you.
        """
        # .with_structured_output() forces the LLM to return valid JSON matching your schema
        structured_llm = self.llm.with_structured_output(schema)
        
        return structured_llm.invoke(prompt)

    def get_stats(self, response) -> Dict:
        """
        LangChain tracks usage metadata automatically
        """
        if hasattr(response, 'response_metadata'):
            return response.response_metadata.get('token_usage', {})
        return {}

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize
    client = LangChainClient(provider="google", model_name="gemini-2.0-flash-exp")

    # 2. Generate Text
    print("--- Text Response ---")
    print(client.generate("Explain GraphRAG in one sentence."))

    # 3. Generate JSON (Structured)
    # Define a simple schema (Pydantic style or raw dict)
    schema = {
        "title": "ProgrammingLanguages",
        "description": "A list of programming languages",
        "type": "object",
        "properties": {
            "languages": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    }
    
    print("\n--- JSON Response ---")
    # This returns a real Python Dict, not a string you have to json.loads()!
    result = client.generate_json("List 3 major AI programming languages", schema)
    print(result)