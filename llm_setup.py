"""
LLM setup and configuration for guidance extraction.
Supports GitHub Models, OpenAI, Anthropic, and Ollama (local).
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.openai import OpenAI


load_dotenv()


def setup_llm(provider="deepseek", model=None, temperature=0.0, timeout=120.0, verbose=True):
    """
    Configure LlamaIndex with the specified LLM provider.
    
    Args:
        provider: "deepseek", "github", "openai", "anthropic", or "ollama"
        model: Specific model name (or None for defaults)
        temperature: 0.0 for deterministic, higher for creative
        timeout: Request timeout in seconds (default: 120.0)
        verbose: Whether to print LLM setup messages (default: True)
    
    Returns:
        Configured LLM instance
    """
    
    if provider == "deepseek":
        api_key_deepseek = os.getenv("DEEPSEEK_API_KEY")
        if not api_key_deepseek:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        
        model = model or "deepseek-chat"  # Fast and free
        
        llm = DeepSeek(
            api_key=api_key_deepseek,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
        if verbose:
            print(f"[LLM] Using DeepSeek: {model} (timeout={timeout}s)")
    
    elif provider == "github":
        api_key_chatGPT = os.getenv("GITHUB_MODELS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key_chatGPT:
            raise ValueError("GITHUB_MODELS_API_KEY or OPENAI_API_KEY not found in .env file")
        
        model = model or "gpt-4o-mini"  # Cheaper, faster for extraction
        
        llm = OpenAI(
            api_key=api_key_chatGPT,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )
        if verbose:
            print(f"[LLM] Using GitHub Models: {model} (timeout={timeout}s)")
    
    elif provider == "openai":
        # Direct OpenAI API - uses separate key from GitHub Models
        api_key = os.getenv("OPENAI_DIRECT_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_DIRECT_API_KEY not found in .env file. Get one from https://platform.openai.com/api-keys")
        
        model = model or "gpt-5"  # Default to gpt-5.2
        
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_base="https://api.openai.com/v1",  # Direct OpenAI endpoint
        )
        if verbose:
            print(f"[LLM] Using OpenAI Direct: {model} (timeout={timeout}s)")
    
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
    
    # Try DeepSeek first (recommended, free)
    try:
        llm = setup_llm(provider="deepseek")
        test_llm(llm)
    except Exception as e:
        print(f"[WARN] DeepSeek setup failed: {e}")
    
  
