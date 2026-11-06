"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Solution for starter-code/template.py. Run the tests with: pytest tests/ -v

SDKs are imported inside each function so that a missing package only breaks
the provider that needs it, and so the tests can mock the client classes.
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# 3 turns = 3 user messages + 3 model messages
MAX_HISTORY_MESSAGES = 6
TABLE_RESPONSE_LIMIT = 50

# Result key -> label shown in the comparison table
MODEL_LABELS = {
    "gpt4o": "GPT-4o",
    "gpt4o_mini": "GPT-4o-Mini",
    "gemini_flash": "Gemini-Flash",
}


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call the OpenAI Chat Completions API.

    Returns (response_text, latency_seconds, usage), where usage has the keys
    'input_tokens' and 'output_tokens'.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response.choices[0].message.content or "", latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call the Google Gemini API using the google-genai SDK.

    Returns (response_text, latency_seconds, usage), where usage has the keys
    'input_tokens' and 'output_tokens'.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count,
    }
    return response.text or "", latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """Call the Anthropic Messages API.

    Returns (response_text, latency_seconds, usage), where usage has the keys
    'input_tokens' and 'output_tokens'.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.content[0].text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def _summarize(model: str, result: tuple[str, float, dict]) -> dict:
    """Turn a call_* result into a comparison entry, including its USD cost."""
    text, latency, usage = result
    rates = PRICING_1M_TOKENS[model]
    input_tokens = usage["input_tokens"]
    output_tokens = usage["output_tokens"]
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
    return {
        "response": text,
        "latency": latency,
        "cost": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def compare_models(prompt: str) -> dict:
    """Send the same prompt to gpt-4o, gpt-4o-mini and gemini-2.5-flash.

    Returns a dict with the keys 'gpt4o', 'gpt4o_mini' and 'gemini_flash'. Each
    value holds response, latency, cost, input_tokens and output_tokens.
    """
    return {
        "gpt4o": _summarize(OPENAI_MODEL, call_openai(prompt)),
        "gpt4o_mini": _summarize(
            OPENAI_MINI_MODEL, call_openai(prompt, model=OPENAI_MINI_MODEL)
        ),
        "gemini_flash": _summarize(GEMINI_MODEL, call_gemini(prompt, model=GEMINI_MODEL)),
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """Run a terminal chatbot that streams Gemini replies as they arrive.

    Keeps the last 3 turns of history. Typing 'quit' or 'exit' ends the session.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    chat_model = os.getenv("GEMINI_CHAT_MODEL") or "gemini-3.8-flash"
    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        user_message = {"role": "user", "parts": [{"text": user_input}]}
        print("Gemini: ", end="", flush=True)
        completed = False
        for attempt in range(4):
            reply_parts = []
            try:
                stream = client.models.generate_content_stream(
                    model=chat_model,
                    contents=history + [user_message],
                )
                for chunk in stream:
                    # Some chunks carry only metadata, so their text is None.
                    if chunk.text:
                        print(chunk.text, end="", flush=True)
                        reply_parts.append(chunk.text)
                print()
                completed = True
                break
            except Exception as exc:
                # Do not replay a stream once text has already been displayed.
                if getattr(exc, "code", None) in (429, 500, 502, 503, 504) and not reply_parts and attempt < 3:
                    delay = 2 * 2**attempt
                    print(f"\nGemini temporarily unavailable; retrying in {delay}s ({attempt + 1}/3)...")
                    time.sleep(delay)
                    print("Gemini: ", end="", flush=True)
                    continue
                print(f"\nRequest failed: {exc}")
                print("Please send your message again, or type 'quit' to exit.")
                break
        # Failed or incomplete turns must not enter the conversation history.
        if not completed or not reply_parts:
            continue

        model_message = {"role": "model", "parts": [{"text": "".join(reply_parts)}]}
        history = (history + [user_message, model_message])[-MAX_HISTORY_MESSAGES:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Call fn(), retrying up to max_retries times on any exception.

    The delay before retry n (counting from 0) is base_delay * 2^n. The last
    exception is re-raised once the retries run out.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception:
            time.sleep(base_delay * 2**attempt)
    return fn()


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """Run compare_models on each prompt, adding the prompt to each result.

    A failed prompt stays in the results with an 'Error: ...' response, so one
    bad request does not lose the rest of the batch.
    """
    results = []
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except Exception as exc:
            comparison = {
                key: {
                    "response": f"Error: {exc}",
                    "latency": 0.0,
                    "cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
                for key in MODEL_LABELS
            }
        results.append({"prompt": prompt, **comparison})
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def _table_cell(text: str) -> str:
    """Keep text on one line and stop pipes from splitting the table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def format_comparison_table(results: list[dict]) -> str:
    """Format batch_compare results as a Markdown table, one row per model."""
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        prompt = _table_cell(result["prompt"])
        for key, label in MODEL_LABELS.items():
            stats = result[key]
            response = " ".join(stats["response"].split())
            if len(response) > TABLE_RESPONSE_LIMIT:
                response = response[:TABLE_RESPONSE_LIMIT] + "..."
            lines.append(
                f"| {prompt} | {label} | {_table_cell(response)} "
                f"| {stats['latency']:.2f}s "
                f"| {stats['input_tokens']}/{stats['output_tokens']} "
                f"| ${stats['cost']:.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # gemini-2.5-flash is closed to new accounts, so live runs can override it in .env.
    GEMINI_MODEL = os.getenv("GEMINI_MODEL") or GEMINI_MODEL

    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
