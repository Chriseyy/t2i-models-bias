"""
run_sd35_cfg_test.py
====================
Spezielles Test-Skript für die Evaluierung von Guidance Scale (CFG)
bei Stable Diffusion 3.5 Large. 
Generiert Bilder über eine CFG-Testreihe bei konstantem Seed.
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
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from dotenv import load_dotenv

# =============================================================
# PFADE FÜR DEN CFG TEST
# =============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
CONFIG_DIR       = PROJECT_ROOT / "config"
OUTPUT_DIR       = PROJECT_ROOT / "outputs"

# EIGENE TRENNORDNER FÜR DEN CFG TEST
IMAGE_DIR        = OUTPUT_DIR / "images_cfg_test" / "sd35"
META_DIR         = OUTPUT_DIR / "metadata_cfg_test" / "sd35"
CHECKPOINT_FILE  = OUTPUT_DIR / "checkpoint_sd35_cfg_test.json"

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# TEST KONFIGURATION
# =============================================================
# Die wissenschaftlich validierte CFG-Testreihe (Symmetrisches Design)
CFG_VALUES = [0.0, 1.0, 2.5, 3.5, 4.0, 7.5, 12.0]

# Exakt 1 Seed für den perfekten, isolierten Vergleich auf deinen Folien
MAX_SEEDS_PER_TEST = 1 

# =============================================================
# LOGGING SETUP
# =============================================================
def setup_logging():
    log_file = OUTPUT_DIR / "generation_sd35_cfg_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("sd35_cfg_test")

# =============================================================
# CONFIG & HELPER
# =============================================================
def load_configs():
    prompt_path = CONFIG_DIR / "prompts.yaml"
    model_path = CONFIG_DIR / "models.yaml"

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)
    with open(model_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    if prompt_cfg is None or model_cfg is None:
        raise ValueError("❌ FEHLER: Config-Dateien konnten nicht geladen werden!")

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
            "model": "sd35_cfg_test"
        }, f, indent=2)

def make_image_id(prompt_id: str, seed: int, cfg: float) -> str:
    # Codiert den CFG-Wert krisensicher direkt in den Bildnamen
    return f"{prompt_id}_seed{seed:03d}_cfg{cfg}"

def save_metadata(image_id, prompt_info, seed, cfg_val, model_cfg, gen_time, image_path):
    gen_cfg = model_cfg["models"]["sd35"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]

    meta = {
        "image_id": image_id,
        "model": "sd35",
        "model_full_name": "Stable Diffusion 3.5 Large",
        "model_id": model_cfg["models"]["sd35"]["model_id"],
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],
        "seed": seed,
        "num_inference_steps": gen_cfg["num_inference_steps"],
        "guidance_scale": cfg_val,  # Der variable CFG-Testwert
        "width": img_cfg["width"],
        "height": img_cfg["height"],
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "note": "SD3.5 CFG Variation Test via Rectified Flow"
    }

    with open(META_DIR / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# =============================================================
# MODELL LADEN (SD 3.5 LARGE)
# =============================================================
def load_model(model_cfg, logger):
    model_id = model_cfg["models"]["sd35"]["model_id"]
    dtype_str = model_cfg["global"]["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    logger.info(f"Lade SD 3.5 Large für den CFG-Test: {model_id}")
    
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.error("❌ Kein HF_TOKEN in der .env gefunden!")
        raise ValueError("Missing Hugging Face Token")

    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token
    )
    pipe = pipe.to("cuda")

    logger.info(f"✅ SD 3.5 Large erfolgreich geladen! VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return pipe

# =============================================================
# GENERIERUNG
# =============================================================
def generate_image(pipe, prompt_info, seed, cfg_val, model_cfg, logger):
    gen_cfg = model_cfg["models"]["sd35"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]
    neg_prompt = gen_cfg.get("negative_prompt_default", "")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    image = pipe(
        prompt=prompt_info["prompt"],
        negative_prompt=neg_prompt if gen_cfg["use_negative_prompt"] else None,
        width=img_cfg["width"],
        height=img_cfg["height"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=cfg_val,  # Überschreibt die Config mit dem Testschritt
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    return image, time.time() - start

# =============================================================
# MAIN PIPELINE EXECUTION
# =============================================================
def main(dry_run=False, resume=True):
    logger = setup_logging()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🔬 STARTE SD3.5 CFG-ABLATION-TEST")
    logger.info(f"Zeitstempel: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    prompt_cfg, model_cfg = load_configs()
    prompts = build_prompt_list(prompt_cfg)
    
    # Isoliert exakt das erste Element der Seed-Liste für das paperkonforme Grid
    seeds = prompt_cfg["seeds"][:MAX_SEEDS_PER_TEST] 
    total_images = len(prompts) * len(seeds) * len(CFG_VALUES)

    logger.info(f"Prompts: {len(prompts)} | Analysierte Seeds: {len(seeds)} | CFG-Stufen: {len(CFG_VALUES)}")
    logger.info(f"Zielmenge: {total_images} Bilder.")

    completed = load_checkpoint() if resume else set()
    if completed:
        logger.info(f"Checkpoint geladen: {len(completed)} Bilder übersprungen.")

    if dry_run:
        logger.info("🔍 Dry-Run beendet.")
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

                logger.info(f"\n📸 Run: {image_id} | CFG: {cfg_val} | Seed: {seed}")

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
                    logger.info(f"   ✅ {gen_time:.1f}s | Fortschritt: {success_count}/{total_images} | ETA: {eta/60:.1f} min")

                except Exception as e:
                    logger.error(f"   ❌ Fehler bei {image_id}: {e}")
                    failed.append({"id": image_id, "error": str(e)})
                    torch.cuda.empty_cache()
                    gc.collect()

    total_time = time.time() - total_start
    logger.info(f"\n🎉 CFG-ABLA-TEST BEENDET! Erfolgreich: {len(completed)}/{total_images} | Fehler: {len(failed)}")

    if failed:
        with open(OUTPUT_DIR / "failed_sd35_cfg_test.json", "w") as f:
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