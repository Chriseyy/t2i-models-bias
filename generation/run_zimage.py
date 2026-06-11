"""
run_zimage.py
==============
Generation-Script für Z-Image (Tongyi-MAI/Z-Image) mit Cross-Lingual Support

Vorteile für die Masterarbeit:
- 6B Parameter Base-Modell (passt locker in die RTX 5090)
- Ideal für Bias-Testing durch den asiatischen/östlichen Ursprung
- Läuft nativ in bfloat16 ohne Komprimierung
- Dynamischer Sprach-Switch via --chinese Flag.
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
from diffusers import ZImagePipeline # <--- Die spezielle Pipeline für das Z-Image Modell
from PIL import Image
from dotenv import load_dotenv

# =============================================================
# PFADE (Standard-Vorgaben für den englischen Haupt-Lauf)
# =============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
CONFIG_DIR      = PROJECT_ROOT / "config"
OUTPUT_DIR      = PROJECT_ROOT / "outputs"

IMAGE_DIR       = OUTPUT_DIR / "images" / "zimage"
META_DIR        = OUTPUT_DIR / "metadata" / "zimage"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_zimage.json"

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# LOGGING
# =============================================================
def setup_logging(chinese=False):
    log_name = "generation_zimage_chines.log" if chinese else "generation_zimage.log"
    log_file = OUTPUT_DIR / log_name
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("zimage")

# =============================================================
# CONFIG & HELPER
# =============================================================
def load_configs(prompt_filename="prompts.yaml"):
    with open(CONFIG_DIR / prompt_filename, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)
    with open(CONFIG_DIR / "models.yaml", "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    return prompt_cfg, model_cfg

def build_prompt_list(prompt_cfg):
    base_template = prompt_cfg["base_template"]
    all_prompts = []
    for category_name, category_data in prompt_cfg["prompts"].items():
        for item in category_data["items"]:
            full_prompt = base_template.format(subject=item["subject"])
            all_prompts.append({
                "id": item["id"],
                "subject": item["subject"],
                "prompt": full_prompt,
                "category": category_name,
                "expected_bias": item.get("expected_bias", "unknown")
            })
    return all_prompts

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()

def save_checkpoint(completed: set, is_chinese=False):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed": list(completed),
            "last_updated": datetime.now().isoformat(),
            "model": "zimage_chines" if is_chinese else "zimage"
        }, f, indent=2)

def make_image_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_seed{seed:03d}"

def save_metadata(image_id, prompt_info, seed, model_cfg, gen_time, image_path, steps, cfg_scale, is_chinese=False):
    zimage_cfg = model_cfg.get("models", {}).get("zimage", {})
    img_cfg = model_cfg.get("global", {}).get("output_size", {"width": 1024, "height": 1024})

    meta = {
        "image_id": image_id,
        "model": "zimage",
        "language": "chinese" if is_chinese else "english",
        "model_full_name": zimage_cfg.get("name", "Tongyi-MAI/Z-Image"),
        "model_id": zimage_cfg.get("model_id", "Tongyi-MAI/Z-Image"),
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "negative_prompt": "bad quality, worst quality, deformed, extra limbs, floating objects, surreal, abstract, artifacts, messy background",
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": cfg_scale,
        "cfg_normalization": True,
        "width": img_cfg.get("width", 1024),
        "height": img_cfg.get("height", 1024),
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "note": "Cross-Lingual Deep Dive (East Base)" if is_chinese else "6B Base Model (East) loaded natively in bf16"
    }

    with open(META_DIR / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# =============================================================
# MODELL LADEN
# =============================================================
def load_model(model_cfg, logger):
    model_id = model_cfg.get("models", {}).get("zimage", {}).get("model_id", "Tongyi-MAI/Z-Image")
    dtype = torch.bfloat16

    logger.info(f"Lade natives Z-Image Modell: {model_id}")
    
    pipe = ZImagePipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    logger.info(f"✅ Z-Image erfolgreich geladen! VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return pipe

# =============================================================
# GENERIERUNG
# =============================================================
def generate_image(pipe, prompt_info, seed, model_cfg, logger):
    zimage_cfg = model_cfg.get("models", {}).get("zimage", {}).get("generation", {})
    steps = zimage_cfg.get("num_inference_steps", 50)
    cfg_scale = zimage_cfg.get("guidance_scale", 4.0)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    logger.info(f"   🖌️ Generiere Bild ({steps} Steps) für: {prompt_info['prompt']}")
    TECHNICAL_NEGATIVE_PROMPT = "bad quality, worst quality, deformed, extra limbs, floating objects, surreal, abstract, artifacts, messy background"

    image = pipe(
        prompt=prompt_info["prompt"],
        negative_prompt=TECHNICAL_NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=cfg_scale,
        cfg_normalization=True,
        generator=generator,
    ).images[0]

    gen_time = time.time() - start
    return image, gen_time, steps, cfg_scale

# =============================================================
# MAIN
# =============================================================
def main(dry_run=False, resume=True, chinese=False):
    global IMAGE_DIR, META_DIR, CHECKPOINT_FILE

    # EXAKTE PFAD-ANPASSUNG FÜR DEINEN CHINESISCHEN OUTPUT
    if chinese:
        IMAGE_DIR       = OUTPUT_DIR / "images_chines" / "zimage"
        META_DIR        = OUTPUT_DIR / "metadata_chines" / "zimage"
        CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_zimage_chines.json"

    logger = setup_logging(chinese=chinese)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"BIAS EVALUATION - Z-IMAGE 6B ({'CHINESE DEEP DIVE' if chinese else 'ENGLISH MAIN'})")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Lade die jeweils korrekte Konfigurationsdatei
    prompt_file = "prompt_chines.yaml" if chinese else "prompts.yaml"
    logger.info(f"Nutze Konfigurationsdatei: config/{prompt_file}")

    prompt_cfg, model_cfg = load_configs(prompt_file)
    prompts = build_prompt_list(prompt_cfg)
    seeds   = prompt_cfg["seeds"]
    total_images = len(prompts) * len(seeds)

    logger.info(f"Prompts: {len(prompts)} | Seeds: {len(seeds)} | Gesamt: {total_images}")

    completed = load_checkpoint() if resume else set()
    if completed:
        logger.info(f"Checkpoint: {len(completed)} bereits fertig")

    if dry_run:
        logger.info("Dry-Run aktiv. Beende Skript vor Modell-Initalisierung.")
        return

    pipe = load_model(model_cfg, logger)
    failed = []
    success_count = len(completed)
    total_start = time.time()

    for prompt_info in prompts:
        for seed in seeds:
            image_id = make_image_id(prompt_info["id"], seed)

            if image_id in completed:
                continue

            logger.info(f"\n📸 {image_id} | Seed {seed}")

            try:
                image, gen_time, steps, cfg_scale = generate_image(pipe, prompt_info, seed, model_cfg, logger)

                image_path = IMAGE_DIR / f"{image_id}.png"
                image.save(image_path, format="PNG")

                save_metadata(image_id, prompt_info, seed, model_cfg, gen_time, image_path, steps, cfg_scale, is_chinese=chinese)

                completed.add(image_id)
                save_checkpoint(completed, is_chinese=chinese)
                success_count += 1

                elapsed = time.time() - total_start
                eta = (elapsed / success_count) * (total_images - success_count) if success_count > 0 else 0
                logger.info(f"   ✅ {gen_time:.2f}s | {success_count}/{total_images} | ETA: {eta/60:.1f} min")

            except Exception as e:
                logger.error(f"   ❌ {image_id}: {e}")
                failed.append({"id": image_id, "error": str(e)})
                torch.cuda.empty_cache()
                gc.collect()

    total_time = time.time() - total_start
    logger.info(f"\n✅ {len(completed)}/{total_images} | ❌ {len(failed)} | ⏱️ {total_time/60:.1f} min")

    fail_filename = "failed_zimage_chines.json" if chinese else "failed_zimage.json"
    if failed:
        with open(OUTPUT_DIR / fail_filename, "w") as f:
            json.dump(failed, f, indent=2)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--chinese", action="store_true", help="Startet den chinesischen Cross-Lingual Deep Dive")
    args = parser.parse_args()
    
    main(dry_run=args.dry_run, resume=not args.no_resume, chinese=args.chinese)