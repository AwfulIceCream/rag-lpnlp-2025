from typing import Optional
import logging
import litellm


# Suppress LiteLLM verbose logging
litellm.set_verbose = False
litellm.suppress_debug_info = True

# Suppress litellm logger
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMClient:
    """
    LLM-agnostic client using LiteLLM.
    
    Supported providers and model formats:
    - Groq: "groq/llama-3.1-8b-instant", "groq/mixtral-8x7b-32768"
    - OpenRouter: "openrouter/google/gemma-2-9b-it:free"
    - OpenAI: "gpt-4o-mini", "gpt-4o"
    - Ollama: "ollama/llama3.2", "ollama/mistral"
    - Anthropic: "claude-3-haiku-20240307"
    """
    
    PROVIDER_EXAMPLES = {
        "groq": ["groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile", "groq/mixtral-8x7b-32768"],
        "openrouter": ["openrouter/google/gemma-2-9b-it:free", "openrouter/meta-llama/llama-3.2-3b-instruct:free"],
        "openai": ["gpt-4o-mini", "gpt-4o"],
        "ollama": ["ollama/llama3.2", "ollama/mistral"],
        "anthropic": ["claude-3-haiku-20240307"],
    }
    
    def __init__(
        self,
        model: str = "groq/llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        """
        Initialize LLM client.
        
        Args:
            model: Model identifier (e.g., "groq/llama-3.1-8b-instant")
            api_key: API key for the provider
            api_base: Optional custom API base URL (for Ollama: http://localhost:11434)
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        
        # Detect provider from model name
        self.provider = self._detect_provider(model)
    
    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        if model.startswith("groq/"):
            return "groq"
        elif model.startswith("openrouter/"):
            return "openrouter"
        elif model.startswith("ollama/"):
            return "ollama"
        elif model.startswith("claude"):
            return "anthropic"
        else:
            return "openai"
    
    def _get_api_key_name(self) -> str:
        """Get environment variable name for API key."""
        key_names = {
            "groq": "GROQ_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "ollama": None,
        }
        return key_names.get(self.provider, "OPENAI_API_KEY")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Prepare kwargs
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        # Add API key if provided
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        # Add API base for Ollama or custom endpoints
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        # Call LiteLLM
        response = litellm.completion(**kwargs)
        
        return response.choices[0].message.content
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: list[tuple[int, str, str]],  # (id, title, text)
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer using retrieved context chunks with citations.
        
        Args:
            query: User's question
            context_chunks: List of (chunk_id, title, text) tuples
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated answer with citations
        """
        # Build context string with numbered sources
        context_parts = []
        for i, (chunk_id, title, text) in enumerate(context_chunks, 1):
            context_parts.append(f"[{i}] {title}\n{text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """You are a helpful recipe assistant. Answer questions about recipes based on the provided context.

IMPORTANT RULES:
1. Use ONLY the information from the provided recipe sources
2. Include citations in your answer using [N] format where N is the source number
3. If the answer cannot be found in the sources, say so
4. Be concise but helpful
5. When describing a recipe, mention key ingredients and steps"""

        user_prompt = f"""Context (Recipe Sources):
{context_str}

Question: {query}

Please provide a helpful answer based on the recipe sources above. Include citations [1], [2], etc. to reference which recipes you're using."""

        return self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )


def get_provider_info() -> dict:
    """Get information about supported providers."""
    return {
        "groq": {
            "name": "Groq",
            "description": "Fast inference, free tier available",
            "api_key_env": "GROQ_API_KEY",
            "models": ["groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile"],
            "signup": "https://console.groq.com/"
        },
        "openrouter": {
            "name": "OpenRouter", 
            "description": "Multiple models, some free options",
            "api_key_env": "OPENROUTER_API_KEY",
            "models": ["openrouter/google/gemma-2-9b-it:free"],
            "signup": "https://openrouter.ai/"
        },
        "openai": {
            "name": "OpenAI",
            "description": "GPT models, paid",
            "api_key_env": "OPENAI_API_KEY",
            "models": ["gpt-4o-mini", "gpt-4o"],
            "signup": "https://platform.openai.com/"
        },
        "ollama": {
            "name": "Ollama",
            "description": "Local models, no API key needed",
            "api_key_env": None,
            "models": ["ollama/llama3.2", "ollama/mistral"],
            "api_base": "http://localhost:11434"
        }
    }

