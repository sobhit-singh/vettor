# Model Providers & Small + Frontier Routing 🧠

Vettor treats model inference as a decoupled, swappable layer. It adopts a **Small + Frontier** routing philosophy:

1. **Small Local Models (SLMs)**: Run high-frequency DOM observation, element identification, and repetitive action replay locally at zero cost and sub-second latency.
2. **Frontier Models**: Escalate complex bug triage, ambiguous error messages, and autonomous planning to cloud models (Claude 3.7, GPT-4o, DeepSeek R1).

---

## Provider Format

Model identifiers follow the standard `provider/model_id` format:

```text
<provider_key>/<model_name_or_path>
```

Examples:
- `openai/gpt-4o-mini`
- `llamacpp/lfm-3-1.7b`
- `ollama/qwen2.5:3b`
- `litellm/anthropic/claude-3-7-sonnet`
- `openrouter/meta-llama/llama-3.3-70b-instruct`
- `vps/my-hosted-vllm-model`

---

## 1. Local `llama-server` (Zero-Config SLM)

Vettor includes an automatic discovery manager for `llama-server`:

- Automatically checks `QA_LLAMA_BIN` and standard system paths (`PATH`, WindowsApps).
- Automatically discovers downloaded `.gguf` weights in `./models/` and `~/.cache/huggingface/hub/`.
- Verifies server health via `/health` before launching.
- Can spawn and manage the background `llama-server` subprocess automatically.

### Configuring in `vettor.yaml`:

```yaml
model:
  active: "llamacpp/liquid-lfm-3b"
  providers:
    llamacpp:
      type: "llamacpp_local"
      base_url: "http://127.0.0.1:8080/v1"
```

To run manually:

```bash
llama-server -m models/LFM-3b-Q4_K_M.gguf --port 8080 --ctx-size 8192
```

---

## 2. Ollama

If you run [Ollama](https://ollama.ai), configure the `ollama` provider:

```yaml
model:
  active: "ollama/qwen2.5:3b"
  providers:
    ollama:
      type: "openai_compat"
      base_url: "http://127.0.0.1:11434/v1"
      api_key: "ollama"
```

Ensure the model is pulled locally:

```bash
ollama pull qwen2.5:3b
```

---

## 3. OpenAI & OpenAI-Compatible Endpoints

Connect directly to OpenAI or any OpenAI-compatible server (vLLM, SGLang, FastChat, remote VPS):

```yaml
model:
  active: "openai/gpt-4o-mini"
  providers:
    openai:
      type: "openai_compat"
      base_url: "https://api.openai.com/v1"
      api_key_env: "OPENAI_API_KEY"

    vps:
      type: "openai_compat"
      base_url: "https://vllm.internal.company.com/v1"
      api_key_env: "INTERNAL_API_KEY"
```

Set your environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 4. LiteLLM & Multi-Cloud Routing

For multi-provider setups (Anthropic, Bedrock, Vertex AI, OpenRouter), use `type: "litellm"`:

```yaml
model:
  active: "litellm/anthropic/claude-3-7-sonnet"
  providers:
    litellm:
      type: "litellm"
      api_key_env: "ANTHROPIC_API_KEY"

    openrouter:
      type: "openai_compat"
      base_url: "https://openrouter.ai/api/v1"
      api_key_env: "OPENROUTER_API_KEY"
```

---

## 5. Strict Local Privacy Lock 🛡️

When testing pre-release staging portals or applications handling sensitive personal data, enable `privacy_lock`:

```yaml
model:
  privacy_lock: true
```

Or toggle it live in the TUI:

```text
> /private
```

When active, Vettor sanitizes all prompts locally before dispatch:
- Emails, phone numbers, and street addresses are replaced with placeholder tokens (`<EMAIL_REDACTED>`).
- Authorization headers, bearer tokens, and session cookies are scrubbed from network traces.
- Password inputs (`input[type="password"]`) are never logged or mirrored.
