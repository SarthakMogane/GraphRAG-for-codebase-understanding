import os
from typing import Dict, Any, Optional ,List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage ,BaseMessage ,AIMessage
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

    def __init__(self, provider: str = "google", model_name: str = "gemini-2.5-flash",temperature: float = 0.0):
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        
        # Factory pattern to pick the right class
        model_class = self.PROVIDER_MAP.get(provider)
        if not model_class:
            raise ValueError(f"Unsupported provider: {provider}")

        # Initialize the model
        # max_retries is built-in! No need for 'tenacity' decorators.
        self.llm: BaseChatModel = model_class(
            model=model_name,
            max_retries=3,
            api_key=self._get_api_key(provider),
            temperature=temperature
        )

    def _get_api_key(self, provider: str) -> str:
        """Helper to fetch keys from env"""
        keys = {
            "google": os.getenv("GOOGLE_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY")
        }
        return keys.get(provider)

    def generate(self, prompt: str = None,
                  system: Optional[str] = None,
                 messages: Optional[list[dict[str ,str]]]= None ,
                 **kwargs) -> str:
        """Standard text generation"""

        langchain_msg:List[BaseMessage] =[]
        if messages:
            for msg in messages:
                role = msg.get('role')
                content = msg.get('content')
                if role == "system":
                    langchain_msg.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_msg.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_msg.append(AIMessage(cotent =content))
        elif prompt :
            if system:
                langchain_msg.append(SystemMessage(content=system))
            langchain_msg.append(HumanMessage(content=prompt))

        else:
            raise ValueError("generate() requires either 'prompt' or 'messages'")
        
        # Invoke the model
        final_llm = self.llm
        if self.temperature is not None:
            final_llm = self.llm.bind(temperature=self.temperature)
        response = final_llm.invoke(messages,**kwargs)
        
        # LangChain standardizes the content for us
        return response

    def generate_json(self, prompt: str, schema: Dict[str, Any],**kwargs) -> Dict:
        """
        Structured Output (The Killer Feature)
        LangChain handles the "json_schema" vs "function_calling" logic for you.
        """
        final_llm = self.llm


        # Google requires parameters like 'temperature' to be inside 'generation_config'
        # if self.provider == "google" and kwargs:
        #     gen_config = {}
            
        #     # Move known parameters into the config dict
        #     if "temperature" in kwargs:
        #         gen_config["temperature"] = kwargs.pop("temperature")
        #     if "max_tokens" in kwargs:
        #         gen_config["max_output_tokens"] = kwargs.pop("max_tokens")
        #     if "top_p" in kwargs:
        #         gen_config["top_p"] = kwargs.pop("top_p")
            
        #     # Bind the config dictionary instead of raw args
        #     if gen_config:
        #         final_llm = self.llm.bind(generation_config=gen_config)

        if kwargs:
            final_llm = self.llm.bind(**kwargs)
        # .with_structured_output() forces the LLM to return valid JSON matching your schema
        structured_llm = final_llm.with_structured_output(schema)
        
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