# CodeLlama Server: Streaming, Caching, Model Fallbacks (OpenAI + Anthropic), Prompt-tracking

Works with: Anthropic, Huggingface, Cohere, TogetherAI, Azure, OpenAI, etc.

[![PyPI Version](https://img.shields.io/pypi/v/litellm.svg)](https://pypi.org/project/litellm/)
[![PyPI Version](https://img.shields.io/badge/stable%20version-v0.1.345-blue?color=green&link=https://pypi.org/project/litellm/0.1.1/)](https://pypi.org/project/litellm/0.1.1/)
![Downloads](https://img.shields.io/pypi/dm/litellm)

## API Endpoints

### `/chat/completions` (POST)

This endpoint is used to generate chat completions for 50+ support LLM API Models. Use llama2, GPT-4, Claude2 etc

#### Input

This API endpoint accepts all inputs in raw JSON and expects the following inputs

- `prompt` (string, optional): Model prompt
- Additional Optional parameters: `temperature`, `functions`, `function_call`, `top_p`, `n`, `stream`. See the full list of supported inputs here: https://litellm.readthedocs.io/en/latest/input/

#### Example JSON body

For claude-2

```json
{
  "prompt": "write me a function to print hello world"
}
```

### Making an API request to the Code-Gen Server

```python
import requests
import json

url = "localhost:4000/chat/completions"

payload = json.dumps({
  "prompt": "write me a function to print hello world"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)

```

### Output [Response Format]

Responses from the server are given in the following format.
All responses from the server are returned in the following format (for all LLM models). More info on output here: https://litellm.readthedocs.io/en/latest/output/

```json
{
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": ".\n\n```\ndef print_hello_world():\n    print(\"hello world\")\n",
                "role": "assistant"
            }
        }
    ],
    "created": 1693279694.6474009,
    "model": "togethercomputer/CodeLlama-34b-Instruct",
    "usage": {
        "completion_tokens": 14,
        "prompt_tokens": 28,
        "total_tokens": 42
    }
}
```

## Installation & Usage

### Running Locally

1. Clone liteLLM repository to your local machine:
   ```
   git clone https://github.com/BerriAI/openai-proxy-ab-testing
   ```
2. Install the required dependencies using pip
   ```
   pip install requirements.txt
   ```
3. Set your LLM API keys
   ```
   os.environ['OPENAI_API_KEY]` = "YOUR_API_KEY"
   or
   set OPENAI_API_KEY in your .env file
   ```
4. Run the server:
   ```
   python main.py
   ```

## Deploying

1. `GCP`, `AWS`, `Azure`
   This project includes a `Dockerfile` allowing you to build and deploy a Docker Project on your providers

# Support / Talk with founders

- [Our calendar 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / +1 (412) 618-6238
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
