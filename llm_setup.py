"""
LLM setup and configuration for guidance extraction.
Supports DeepSeek, Gemini, OpenAI, Anthropic, GitHub Models.
"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings

load_dotenv()


def setup_llm(provider="deepseek", model=None, temperature=0.0, timeout=120.0, verbose=True):
    """
    Configure LlamaIndex with the specified LLM provider.

    Providers:
        "deepseek"   — DeepSeek V3 (default, cheap, strong extraction)
        "gemini"     — Gemini 2.5 Flash (★ best for elaborate prompts / long context)
        "openai"     — OpenAI direct (GPT-4.1 mini recommended)
        "anthropic"  — Claude Haiku 4.5 (fastest Claude, good throughput)
        "github"     — GitHub Models free tier (GPT-4o family)

    Args:
        provider:    One of the providers listed above.
        model:       Override the default model for the provider.
        temperature: 0.0 for deterministic extraction.
        timeout:     Request timeout in seconds.
        verbose:     Print setup info.

    Returns:
        Configured LLM instance (also set as LlamaIndex global default).
    """

    if provider == "deepseek":
        from llama_index.llms.deepseek import DeepSeek

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not set in .env")

        model = model or "deepseek-chat"
        llm = DeepSeek(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

    elif provider == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env  — get one at https://aistudio.google.com/app/apikey")

        model = model or "gemini-2.5-flash"
        llm = GoogleGenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )

    elif provider == "openai":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("OPENAI_DIRECT_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_DIRECT_API_KEY not set in .env  — get one at https://platform.openai.com/api-keys")

        model = model or "gpt-4.1-mini"
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_base="https://api.openai.com/v1",
        )

    elif provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env  — get one at https://console.anthropic.com/settings/keys")

        model = model or "claude-haiku-4-5-20251001"
        llm = Anthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
        )

    elif provider == "github":
        from llama_index.llms.openai import OpenAI

        api_key = os.getenv("GITHUB_MODELS_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("GITHUB_MODELS_API_KEY not set in .env  — get a token at https://github.com/settings/tokens")

        model = model or "gpt-4o-mini"
        llm = OpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_base="https://models.inference.ai.azure.com",
        )

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Use 'deepseek', 'gemini', 'openai', 'anthropic', or 'github'."
        )

    if verbose:
        print(f"[LLM] provider={provider}  model={model}  temperature={temperature}  timeout={timeout}s")

    Settings.llm = llm
    return llm


def test_llm(llm):
    """Quick connectivity test."""
    print("\n[TEST] Testing LLM connection...")
    response = llm.complete("Say 'Hello, I am working!' in exactly 5 words.")
    print(f"[TEST] Response: {response.text}")
    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test an LLM provider")
    parser.add_argument("--provider", default="deepseek", help="Provider to test")
    parser.add_argument("--model", default=None, help="Model override")
    args = parser.parse_args()

    llm = setup_llm(provider=args.provider, model=args.model)
    test_llm(llm)
