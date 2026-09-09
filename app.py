import io
import sys
from pathlib import Path
from datetime import datetime

from PIL import Image

from fastapi import (
    FastAPI,
    Form,
    Request,
    UploadFile,
    File,
)

from fastapi.responses import (
    Response,
    HTMLResponse,
)

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from transformers import CLIPTokenizer


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SD_DIR = BASE_DIR / "sd"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TEMPLATE_DIR = BASE_DIR / "templates"

MODEL_FILE = (
    BASE_DIR
    / "data"
    / "v1-5-pruned-emaonly.ckpt"
)

VOCAB_FILE = (
    BASE_DIR
    / "data"
    / "vocab.json"
)

MERGES_FILE = (
    BASE_DIR
    / "data"
    / "merges.txt"
)


# Make sd/*.py importable
sys.path.insert(
    0,
    str(SD_DIR)
)

import model_loader
import pipeline


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Stable Diffusion API",
    description="Custom Stable Diffusion Text-to-Image API",
)


# Serve generated images
app.mount(
    "/outputs",
    StaticFiles(
        directory=str(OUTPUT_DIR)
    ),
    name="outputs",
)


# HTML templates
templates = Jinja2Templates(
    directory=str(TEMPLATE_DIR)
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"Using device: {DEVICE}"
)


# --------------------------------------------------
# Load tokenizer and models once
# --------------------------------------------------

print(
    "Loading tokenizer..."
)

tokenizer = CLIPTokenizer(
    str(VOCAB_FILE),
    merges_file=str(MERGES_FILE),
)


print(
    "Loading Stable Diffusion models..."
)

models = (
    model_loader
    .preload_models_from_standard_weights(
        str(MODEL_FILE),
        DEVICE,
    )
)

print(
    "Models loaded:",
    models.keys(),
)


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "Stable Diffusion API is running",
    }


# --------------------------------------------------
# Web app
# --------------------------------------------------

@app.get(
    "/app",
    response_class=HTMLResponse,
)
async def web_app(
    request: Request,
):

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


# --------------------------------------------------
# Gallery
# --------------------------------------------------

@app.get("/gallery")
def gallery():

    images = sorted(
        OUTPUT_DIR.glob("*.png"),
        key=lambda path:
            path.stat().st_mtime,
        reverse=True,
    )

    return {
        "images": [
            image.name
            for image in images
        ]
    }


# --------------------------------------------------
# Generate image
# --------------------------------------------------

@app.post(
    "/generate",
    response_class=Response,
    responses={
        200: {
            "content": {
                "image/png": {}
            },
            "description":
                "Generated PNG image",
        }
    },
)
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(5),
    cfg_scale: float = Form(8.0),
    seed: int = Form(42),
    strength: float = Form(0.9),
    input_image: UploadFile | None = File(None),
):

    # CPU-friendly limits
    steps = max(
        1,
        min(steps, 10),
    )

    cfg_scale = max(
        1.0,
        min(cfg_scale, 14.0),
    )

    strength = max(
        0.1,
        min(strength, 1.0),
    )


    # ----------------------------------------------
    # Optional image-to-image input
    # ----------------------------------------------

    pil_input_image = None

    if input_image is not None:

        image_bytes = (
            await input_image.read()
        )

        pil_input_image = (
            Image.open(
                io.BytesIO(
                    image_bytes
                )
            )
            .convert("RGB")
        )


    # ----------------------------------------------
    # Generate
    # ----------------------------------------------

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=negative_prompt,

        # IMPORTANT:
        input_image=pil_input_image,

        # IMPORTANT:
        strength=strength,

        do_cfg=True,
        cfg_scale=cfg_scale,

        sampler_name="ddim",
        n_inference_steps=steps,

        seed=seed,

        models=models,

        device=DEVICE,
        idle_device="cpu",

        tokenizer=tokenizer,
    )


    # ----------------------------------------------
    # Convert to PIL
    # ----------------------------------------------

    image = Image.fromarray(
        output_image
    )


    # ----------------------------------------------
    # Save to outputs/
    # ----------------------------------------------

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S_%f"
        )
    )

    filename = (
        f"generated_"
        f"{timestamp}_"
        f"{seed}.png"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    image.save(
        output_path,
        format="PNG",
    )

    print(
        f"Image saved: {output_path}"
    )


    # ----------------------------------------------
    # Return image to browser
    # ----------------------------------------------

    image_bytes = io.BytesIO()

    image.save(
        image_bytes,
        format="PNG",
    )

    image_bytes.seek(0)

    return Response(
        content=image_bytes.getvalue(),
        media_type="image/png",
    )