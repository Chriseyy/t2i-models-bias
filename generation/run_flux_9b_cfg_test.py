"""
run_flux_9b_cfg_test.py
====================
Spezielles Test-Skript für die Evaluierung von Guidance Scale (CFG)
bei FLUX.2-klein-9B. 
Generiert nur 5 Bilder pro Prompt/CFG-Kombination und speichert sie separiert.
"""

import os
import sys
import json
import time
import logging
import argparse
import gc
from datetime import datetime
from pathlib import Path

import yaml
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image
from dotenv import load_dotenv

# =============================================================
# PFADE FÜR DEN CFG TEST
# =============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
CONFIG_DIR      = PROJECT_ROOT / "config"
OUTPUT_DIR      = PROJECT_ROOT / "outputs"

# EIGENE ORDNER FÜR DEN TEST
IMAGE_DIR       = OUTPUT_DIR / "images_cfg_test" / "flux_klein"
META_DIR        = OUTPUT_DIR / "metadata_cfg_test" / "flux_klein"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_flux_cfg_test.json"

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# TEST KONFIGURATION
# =============================================================
# Deine gewünschten CFG-Werte zum Testen
# CFG_VALUES = [1.0, 2.5, 6.0, 10.0]
CFG_VALUES = [0.0, 1.0, 2.5, 3.5, 4.0, 7.5, 12.0]
# Anzahl der Bilder pro Prompt und CFG-Wert
MAX_SEEDS_PER_TEST = 1 

# =============================================================
# LOGGING
# =============================================================
def setup_logging():
    log_file = OUTPUT_DIR / "generation_flux_cfg_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("flux_cfg_test")

# =============================================================
# CONFIG & HELPER
# =============================================================
def load_configs():
    with open(CONFIG_DIR / "prompts.yaml", "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "models.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return prompt_cfg, model_cfg

def build_prompt_list(prompt_cfg):
    base_template = prompt_cfg["base_template"]
    all_prompts = []
    for category_name, category_data in prompt_cfg["prompts"].items():
        for item in category_data["items"]:
            all_prompts.append({
                "id": item["id"],
                "subject": item["subject"],
                "prompt": base_template.format(subject=item["subject"]),
                "category": category_name,
                "expected_bias": item.get("expected_bias", "unknown")
            })
    return all_prompts

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()

def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed": list(completed),
            "last_updated": datetime.now().isoformat(),
            "model": "flux_klein_cfg_test"
        }, f, indent=2)

def make_image_id(prompt_id: str, seed: int, cfg: float) -> str:
    # Fügt den CFG Wert in den Dateinamen ein
    return f"{prompt_id}_seed{seed:03d}_cfg{cfg}"

def save_metadata(image_id, prompt_info, seed, cfg_val, model_cfg, gen_time, image_path):
    gen_cfg = model_cfg["models"]["flux_klein"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]

    meta = {
        "image_id": image_id,
        "model": "flux_klein",
        "model_full_name": model_cfg["models"]["flux_klein"]["name"],
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],
        "seed": seed,
        "num_inference_steps": gen_cfg["num_inference_steps"],
        "guidance_scale": cfg_val, # Hier speichern wir den variablen CFG Wert
        "width": img_cfg["width"],
        "height": img_cfg["height"],
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "note": "CFG Variation Test"
    }

    with open(META_DIR / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# =============================================================
# MODELL LADEN
# =============================================================
def load_model(model_cfg, logger):
    cfg = model_cfg["models"]["flux_klein"]
    model_id = cfg["model_id"]
    use_offload = cfg["optimizations"]["enable_model_cpu_offload"]
    dtype = torch.bfloat16

    logger.info(f"Lade natives FLUX.2-klein (9B) Modell: {model_id}")
    
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    
    if use_offload:
        pipe.enable_model_cpu_offload()
        logger.info("   CPU-Offload aktiviert")
    else:
        pipe.to("cuda")
        logger.info("   CPU-Offload deaktiviert")

    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"✅ FLUX.2-klein erfolgreich geladen! VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return pipe

# =============================================================
# GENERIERUNG
# =============================================================
def generate_image(pipe, prompt_info, seed, cfg_val, model_cfg, logger):
    gen_cfg = model_cfg["models"]["flux_klein"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    image = pipe(
        prompt=prompt_info["prompt"], 
        width=img_cfg["width"],
        height=img_cfg["height"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=cfg_val, # Nutzt den variablen CFG Wert
        generator=generator,
    ).images[0]

    return image, time.time() - start

# =============================================================
# MAIN
# =============================================================
def main(dry_run=False, resume=True):
    logger = setup_logging()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CFG TEST - FLUX.2-klein-9B")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    prompt_cfg, model_cfg = load_configs()
    prompts = build_prompt_list(prompt_cfg)
    
    # Schneidet die Seeds auf die ersten 5 ab (oder weniger, falls weniger konfiguriert sind)
    seeds = prompt_cfg["seeds"][:MAX_SEEDS_PER_TEST] 
    
    total_images = len(prompts) * len(seeds) * len(CFG_VALUES)

    logger.info(f"Prompts: {len(prompts)} | Seeds: {len(seeds)} | CFG-Werte: {len(CFG_VALUES)}")
    logger.info(f"Gesamt zu generierende Bilder: {total_images}")

    completed = load_checkpoint() if resume else set()
    if completed:
        logger.info(f"Checkpoint: {len(completed)} bereits fertig")

    if dry_run:
        return

    pipe = load_model(model_cfg, logger)
    failed = []
    success_count = len(completed)
    total_start = time.time()

    for prompt_info in prompts:
        for cfg_val in CFG_VALUES:
            for seed in seeds:
                image_id = make_image_id(prompt_info["id"], seed, cfg_val)

                if image_id in completed:
                    continue

                logger.info(f"\n📸 {image_id} | CFG {cfg_val} | Seed {seed}")

                try:
                    image, gen_time = generate_image(pipe, prompt_info, seed, cfg_val, model_cfg, logger)

                    image_path = IMAGE_DIR / f"{image_id}.png"
                    image.save(image_path, format="PNG")

                    save_metadata(image_id, prompt_info, seed, cfg_val, model_cfg, gen_time, image_path)

                    completed.add(image_id)
                    save_checkpoint(completed)
                    success_count += 1

                    elapsed = time.time() - total_start
                    eta = (elapsed / success_count) * (total_images - success_count) if success_count > 0 else 0
                    logger.info(f"   {gen_time:.2f}s | {success_count}/{total_images} | ETA: {eta/60:.1f} min")

                except Exception as e:
                    logger.error(f"   {image_id}: {e}")
                    failed.append({"id": image_id, "error": str(e)})
                    torch.cuda.empty_cache()
                    gc.collect()

    total_time = time.time() - total_start
    logger.info(f"\n{len(completed)}/{total_images} | ❌ {len(failed)} | ⏱️ {total_time/60:.1f} min")

    if failed:
        with open(OUTPUT_DIR / "failed_flux_cfg_test.json", "w") as f:
            json.dump(failed, f, indent=2)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run, resume=not args.no_resume)