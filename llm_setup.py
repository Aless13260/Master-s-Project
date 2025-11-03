"""
LLM setup and configuration for guidance extraction.
Supports GitHub Models, OpenAI, Anthropic, and Ollama (local).
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI


load_dotenv()


def setup_llm(provider="github", model=None, temperature=0.0):
    """
    Configure LlamaIndex with the specified LLM provider.
    
    Args:
        provider: "github", "openai", "anthropic", or "ollama"
        model: Specific model name (or None for defaults)
        temperature: 0.0 for deterministic, higher for creative
    
    Returns:
        Configured LLM instance
    """
    
    if provider == "github":
        # GitHub Models uses OpenAI-compatible API
        api_key = os.getenv("GITHUB_TOKEN")
        if not api_key:
            raise ValueError("GITHUB_TOKEN not found in .env file")
        
        # Available models: gpt-4o, gpt-4o-mini, meta-llama-3.1-405b-instruct, etc.
        model = model or "gpt-4o-mini"  # Fast and free
        
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            api_base="https://models.inference.ai.azure.com",
            api_version=None,  # GitHub Models doesn't use versioning
        )
        print(f"[LLM] Using GitHub Models: {model}")
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        model = model or "gpt-4o-mini"  # Cheaper, faster for extraction
        # model = model or "gpt-4o"  # Use this for higher quality
        
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
        print(f"[LLM] Using OpenAI: {model}")
    
    
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'github', 'openai', 'anthropic', or 'ollama'")
    
    # Set as global default for LlamaIndex
    Settings.llm = llm
    
    return llm


def test_llm(llm):
    """Quick test to verify LLM is working."""
    print("\n[TEST] Testing LLM connection...")
    
    response = llm.complete("Say 'Hello, I am working!' in exactly 5 words.")
    print(f"[TEST] Response: {response.text}")
    
    return response


if __name__ == "__main__":
    # Test different providers
    print("=== LLM Setup Test ===\n")
    
    # Try GitHub Models first (recommended, free)
    try:
        llm = setup_llm(provider="github", model="gpt-5-mini")
        test_llm(llm)
    except Exception as e:
        print(f"[WARN] GitHub Models setup failed: {e}")
    
  
