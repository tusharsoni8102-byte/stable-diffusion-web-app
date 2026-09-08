# Stable Diffusion Web Application

A custom Stable Diffusion web application built with FastAPI, PyTorch, and Docker.

The application provides a web interface for generating images using a custom Stable Diffusion model.

## Features

- Text-to-image generation
- Image-to-image generation
- Negative prompts
- Adjustable inference steps
- CFG scale control
- Seed control
- Image strength control
- Generated image download
- Generation history/gallery
- FastAPI backend
- Docker support

---

# Project Structure

```text
pytorch-stable-diffusion/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── data/
│   ├── vocab.json
│   ├── merges.txt
│   └── v1-5-pruned-emaonly.ckpt
│
├── templates/
│   └── index.html
│
├── sd/
│   ├── pipeline.py
│   ├── model_loader.py
│   ├── ddim.py
│   └── ...
│
└── outputs/
    └── Generated images