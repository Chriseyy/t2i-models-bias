
# Evaluation of Bias and Fairness in Text-to-Image Models

> **Official Repository for the Paper:** *"Evaluierung von Bias und Fairness in aktuellen Text-zu-Bild-Modellen"*

This repository contains the complete end-to-end automated pipeline used to generate, process, and analyze images for demographic bias (gender, ethnicity) and fairness in state-of-the-art Text-to-Image (T2I) models. It serves as the codebase and documentation for the methodological approach described in the paper.

## Key Features

* **Multi-Model Generation:** Scripts to systematically generate images using Western (SD 3.5, FLUX.2) and Asian (Qwen-Image, Z-Image) models in both FP16 and 4-bit quantized formats.
* **Automated Processing:** YOLOv11-based facial detection and automated cropping (`yolo11n.pt`).
* **Multi-MLLM-as-a-Judge:** Automated phenotype classification using a local Multi-MLLM ensemble via Ollama (Qwen2.5-VL, InternVL3.5, Gemma4).
* **Latent Space Analysis:** Extraction of CLIP and DINOv3 embeddings with UMAP/t-SNE dimensionality reduction to calculate Silhouette Scores.
* **CFG Ablation Studies:** Dedicated pipeline to test the architectural dynamics of Classifier-Free Guidance (CFG) scaling.

---

## Repository Structure

Based on the experimental setup of the paper, the codebase is strictly separated into configuration, generation, and advanced analysis logic:

```text
T2I-MODELS-BIAS/
├── analysis/                          # Advanced Evaluation, Visualization & Metrics
│   ├── cluster_and_visualize.py       # Dimensionality reduction (t-SNE/UMAP) & clustering of latent spaces
│   ├── create_cfg_plots.py            # Generates comparison matrices for CFG ablation studies
│   ├── crop_persons.py                # YOLOv11-based automated face/person cropping with smart-skip
│   ├── deepface_analyse.py            # CV baseline evaluation (Gender, Race) + FastNet Anti-Spoofing check
│   ├── embedding_metrics.py           # Calculates Silhouette Scores & Inter-Model Cosine Distances
│   ├── evaluate_results.py            # Master evaluation script (Macro & Fair Comparison, divergence plots, heatmaps)
│   ├── extract_clip_embeddings.py     # Extracts CLIP vectors (ViT-Large) from cropped images
│   ├── human_eval_tool.py             # Custom Tkinter GUI for manual annotation (incl. Monk Skin Tone Pipette)
│   ├── vlm_divergenz_analyse.py       # Computes Inter-Rater Reliability & "Lone-Wolf" effects among VLMs
│   └── ...                            # Additional metric add-ons and prompt dashboard generators
├── config/                 # YAML configuration files
│   ├── models.yaml                    # Model paths, architectures, and inference parameters
│   └── prompts.yaml                   # Stereotype-prone occupational prompts
├── generation/             # Image generation scripts per model
│   ├── run_flux.py / run_flux_9b.py
│   ├── run_qwen.py / run_sd35.py / run_zimage.py
│   └── *_cfg_test.py                  # Specific CFG ablation runners
├── outputs/                # Generated artifacts 
│   ├── cropped_persons/ / images/
│   ├── metadata/ / plots/
│   └── step_test/                     # .pkl embeddings and .json checkpoints
├── paper/                  # LaTeX source files for the publication
├── pipeline.py             # Main entry point for the Makro-Bias pipeline
└── pipeline_cfg.py         # Main entry point for the CFG-Ablation pipeline

```

---

## Models Evaluated

**Generative T2I Models:**

* Stable Diffusion 3.5 Large (Western, FP16)
* FLUX.2 Klein / Dev (Western, FP16 / 4-bit)
* Z-Image (Asian, FP16)
* Qwen-Image (Asian, 4-bit)

**MLLM Evaluators (via Ollama):**

* Qwen2.5-VL
* InternVL3.5
* Gemma4:e4b

---
