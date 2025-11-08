"""
Backend abstraction layer for Parley chat application.

Supports OpenAI, Anthropic, and Gemini backends with clean async interfaces.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any
import asyncio


class Backend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: dict[str, Any]):
        """Initialize backend with configuration."""
        self.config = config
        self.model = config.get("model", "")

    @abstractmethod
    async def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Tuple of (response_content, usage_dict)
        """
        pass

    @abstractmethod
    async def stream_chat(self, messages: list[dict], callback: Callable[[str], None]) -> tuple[str, dict]:
        """
        Stream a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            callback: Function called with each chunk of text

        Returns:
            Tuple of (full_response_content, usage_dict)
        """
        pass


class OpenAIBackend(Backend):
    """OpenAI-compatible backend (works with local servers too)."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        from openai import AsyncOpenAI

        base_url = config.get("base_url", "https://api.openai.com/v1")
        api_key = config.get("api_key", "not-needed")

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = config.get("model", "gpt-3.5-turbo")

    async def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send non-streaming chat request."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=120.0
        )

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return content, usage

    async def stream_chat(self, messages: list[dict], callback: Callable[[str], None]) -> tuple[str, dict]:
        """Stream chat request with callback for each chunk."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            timeout=120.0
        )

        full_content = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_content += text
                callback(text)

        # Approximate token count (real impl would track properly)
        usage = {
            "prompt_tokens": sum(len(m["content"].split()) for m in messages),
            "completion_tokens": len(full_content.split()),
            "total_tokens": sum(len(m["content"].split()) for m in messages) + len(full_content.split())
        }

        return full_content, usage


class AnthropicBackend(Backend):
    """Anthropic Claude backend."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        import anthropic

        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = config.get("model", "claude-3-5-sonnet-20241022")

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] in ("user", "assistant"):
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        return system_message, anthropic_messages

    async def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send non-streaming chat request."""
        system_msg, anthropic_msgs = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": 4096
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = await self.client.messages.create(**kwargs)

        content = response.content[0].text
        usage = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

        return content, usage

    async def stream_chat(self, messages: list[dict], callback: Callable[[str], None]) -> tuple[str, dict]:
        """Stream chat request with callback for each chunk."""
        system_msg, anthropic_msgs = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": anthropic_msgs,
            "max_tokens": 4096
        }
        if system_msg:
            kwargs["system"] = system_msg

        full_content = ""

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                full_content += text
                callback(text)

            # Get final message for usage stats
            message = await stream.get_final_message()
            usage = {
                "prompt_tokens": message.usage.input_tokens,
                "completion_tokens": message.usage.output_tokens,
                "total_tokens": message.usage.input_tokens + message.usage.output_tokens
            }

        return full_content, usage


class GeminiBackend(Backend):
    """Google Gemini backend."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        from google import genai

        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Gemini API key required")

        self.client = genai.Client(api_key=api_key)
        self.model = config.get("model", "gemini-2.0-flash-exp")

    def _convert_messages(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format."""
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"Human: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(parts)
        if messages and messages[-1]["role"] == "user":
            prompt += "\n\nAssistant:"

        return prompt

    async def chat(self, messages: list[dict]) -> tuple[str, dict]:
        """Send non-streaming chat request."""
        prompt = self._convert_messages(messages)

        # Run sync API in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
        )

        content = response.text if hasattr(response, 'text') else str(response)

        # Approximate token count (Gemini uses chars/4)
        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(content) // 4,
            "total_tokens": (len(prompt) + len(content)) // 4
        }

        return content, usage

    async def stream_chat(self, messages: list[dict], callback: Callable[[str], None]) -> tuple[str, dict]:
        """Stream chat request with callback for each chunk."""
        prompt = self._convert_messages(messages)

        full_content = ""

        # Stream API is also sync, so we need to handle it carefully
        loop = asyncio.get_event_loop()

        # Generator wrapper for streaming
        def stream_gen():
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt
            )

        stream = await loop.run_in_executor(None, stream_gen)

        for chunk in stream:
            if hasattr(chunk, 'text'):
                text = chunk.text
                full_content += text
                callback(text)

        usage = {
            "prompt_tokens": len(prompt) // 4,
            "completion_tokens": len(full_content) // 4,
            "total_tokens": (len(prompt) + len(full_content)) // 4
        }

        return full_content, usage


def create_backend(config: dict[str, Any]) -> Backend:
    """Factory function to create appropriate backend from config."""
    backend_type = config.get("backend", "openai").lower()

    if backend_type == "openai":
        return OpenAIBackend(config)
    elif backend_type == "anthropic":
        return AnthropicBackend(config)
    elif backend_type == "gemini":
        return GeminiBackend(config)
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
