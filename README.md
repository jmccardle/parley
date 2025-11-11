# Parley

![Parley Chat Interface](https://i.imgur.com/mqO7fKT.png)

A minimalist, performant chat interface for LLMs. Clean, simple, fast.

## Features

- **Multi-backend support**: OpenAI, Anthropic, Gemini (including local LLM servers)
- **Performance-first streaming**: 30Hz throttled updates, no CPU fan spin-up
- **Clean UI**: Textual-based interface with command palette
- **File-based persistence**: Each chat saved as JSON
- **History navigation**: Up/Down arrows in input
- **Incremental rendering**: No reactive thrashing, messages mount incrementally

## Screenshots

<a href="https://i.imgur.com/wd29JP8.png"><img src="https://i.imgur.com/wd29JP8.png" width="400" alt="Editing system prompt"></a>

*Editing system prompt*

<a href="https://i.imgur.com/EuVcu8s.png"><img src="https://i.imgur.com/EuVcu8s.png" width="400" alt="Custom system prompt response"></a>

*Response with edited system prompt*

## Installation

```bash
cd parley
pip install textual openai anthropic google-generativeai
```

## Configuration

Edit `config.json` to set up your models and API keys:

```json
{
  "models": {
    "local-llm": {
      "backend": "openai",
      "model": "your-model-name",
      "base_url": "http://localhost:8000/v1",
      "api_key": "not-needed"
    }
  },
  "default_model": "local-llm",
  "system_prompt": "You are a helpful assistant."
}
```

## Usage

```bash
python chat_app.py
```

### Keyboard Shortcuts

- **Ctrl+N**: New chat
- **Ctrl+P**: Command palette (fuzzy search all commands)
- **Ctrl+B**: Toggle sidebar
- **Up/Down**: Navigate input history
- **Ctrl+C**: Quit

### Command Palette

Press **Ctrl+P** to access:
- New Chat with specific model
- Clear current chat
- Export chat to markdown

## Architecture

- `chat_app.py` (~500 lines) - Main Textual application
- `backends.py` (~300 lines) - Backend abstraction layer
- `config.json` - Model and API key configuration
- `chats/` - Saved conversation files

## Design Principles

1. **No reactive recompose**: Messages mount incrementally, never rebuild entire chat
2. **Throttled streaming**: 30Hz max updates, prevents UI thrashing
3. **Clean abstractions**: Backend protocol is simple: `chat()` and `stream_chat()`
4. **File-based state**: One JSON per chat, no database complexity
5. **Extensible**: Easy to add new backends or features without refactoring

## Extension Points

- **Add backends**: Inherit from `Backend` class in `backends.py`
- **Add features**: System command palette makes features discoverable
- **Add agents**: Messages are dicts, easy to add `{"role": "tool", ...}` later

## Performance Notes

Streaming is throttled to 30Hz (33ms between renders) to prevent:
- CPU fan spin-up from rapid UI updates
- Textual event loop saturation
- Perceived latency (30Hz is imperceptible to humans)

This makes streaming feel smooth without the performance cost.
