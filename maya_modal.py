# modal_app.py
import modal

app = modal.App("sub1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # 1. Install runtime deps. You can trim this, but this mirrors their stack.
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "torchaudio>=2.5.0",
        "transformers>=4.57.0",
        "accelerate>=1.10.0",
        "vllm>=0.11.0",
        "xformers>=0.0.32",
        "snac>=1.2.1",
        "soundfile>=0.13.0",
        "numpy>=2.1.0",
        "librosa>=0.11.0",
        "scipy>=1.15.0",
        "fastapi>=0.119.0",
        "uvicorn[standard]>=0.38.0",
        "pydantic>=2.12.0",
        "pydantic-settings>=2.11.0",
        "python-multipart>=0.0.20",
        "httpx>=0.28.0",
        "python-dotenv>=1.1.0",
        "huggingface-hub>=0.35.0",
        "tqdm>=4.67.0",
        "openai>=2.5.0",
        "python-Levenshtein>=0.21.0",
    )
    # 2. This is the crucial part: include the local `maya1` Python package.
    #   Because modal_app.py lives in the same directory as `maya1/`,
    #   Python can import `maya1` locally, and Modal will copy that package
    #   into /root/maya1 inside the container.
    .add_local_python_source("maya1")
)


@app.function(
    image=image,
    gpu="A10G",       # or "L4", "T4", etc.
    timeout=600,
    min_containers=1,
    max_containers=1,
)
@modal.asgi_app()
def maya1_tts_app():
    """
    Runs inside the Modal container.

    - Loads env (.env) if present
    - Sets default model path
    - Imports FastAPI `app` from maya1.api_v2
    - Returns it as the ASGI app
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # If not overridden, use HF repo as model path
    os.environ.setdefault("MAYA1_MODEL_PATH", "maya-research/maya1")

    # If the model is gated and you need auth, you can also set:
    # os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", "<your_hf_token>")

    from maya1.api_v2 import app as fastapi_app
    return fastapi_app
