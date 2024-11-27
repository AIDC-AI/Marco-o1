# Deploy Marco-o1 API with FastAPI

This example provides an API using FastAPI to interact with a language model. You have the option to choose between using streaming responses or non-streaming responses, depending on your use-case requirements.

## Requirements

- FastAPI
- Uvicorn
- Transformers
- Torch
- VLLM
- HTTPX (for streaming response)
- Requests (optional, for non-streaming)


## Running the API Server

### Non-Streaming Mode

To start the FastAPI server with non-streaming responses:

```bash
uvicorn vllm_fastapi:app --workers 1
```

To run a client with non-streaming responses:

```bash
python3 client.py
```

### Streaming Mode

To start the FastAPI server with non-streaming responses:

```bash
uvicorn stream_vllm_fastapi:app --workers 1
```

To run a client with non-streaming responses:

```bash
python3 stream_client.py
```
