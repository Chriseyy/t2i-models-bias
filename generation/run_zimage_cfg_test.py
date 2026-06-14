"""
run_zimage_cfg_test.py
======================
Spezielles Test-Skript für die Evaluierung von Guidance Scale (CFG)
bei Z-Image (Tongyi-MAI/Z-Image 6B). 
Generiert Bilder über die CFG-Testreihe bei konstantem, isoliertem Seed.
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
from diffusers import ZImagePipeline  # <--- Spezifische Z-Image Pipeline
from PIL import Image
from dotenv import load_dotenv

# =============================================================
# PFADE FÜR DEN CFG TEST
# =============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
CONFIG_DIR       = PROJECT_ROOT / "config"
OUTPUT_DIR       = PROJECT_ROOT / "outputs"

# EIGENE TRENNORDNER FÜR DEN Z-IMAGE CFG TEST
IMAGE_DIR        = OUTPUT_DIR / "images_cfg_test" / "zimage"
META_DIR         = OUTPUT_DIR / "metadata_cfg_test" / "zimage"
CHECKPOINT_FILE  = OUTPUT_DIR / "checkpoint_zimage_cfg_test.json"

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# TEST KONFIGURATION
# =============================================================
# Symmetrisches Test-Design (4.0 ist deine Baseline aus der Haupt-Pipeline)
CFG_VALUES = [0.0, 1.0, 2.5, 3.5, 4.0, 7.5, 12.0]

# Exakt 1 Seed für den perfekten Vorher-Nachher-Vergleich ohne Layout-Verschiebung
MAX_SEEDS_PER_TEST = 1 

# =============================================================
# LOGGING
# =============================================================
def setup_logging():
    log_file = OUTPUT_DIR / "generation_zimage_cfg_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("zimage_cfg_test")

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
            "model": "zimage_cfg_test"
        }, f, indent=2)

def make_image_id(prompt_id: str, seed: int, cfg: float) -> str:
    # Codiert den CFG-Wert krisensicher direkt in den Bildnamen
    return f"{prompt_id}_seed{seed:03d}_cfg{cfg}"

def save_metadata(image_id, prompt_info, seed, cfg_val, model_cfg, gen_time, image_path, steps):
    zimage_cfg = model_cfg.get("models", {}).get("zimage", {})
    img_cfg = model_cfg.get("global", {}).get("output_size", {"width": 1024, "height": 1024})

    meta = {
        "image_id": image_id,
        "model": "zimage",
        "model_full_name": zimage_cfg.get("name", "Tongyi-MAI/Z-Image"),
        "model_id": zimage_cfg.get("model_id", "Tongyi-MAI/Z-Image"),
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "negative_prompt": "bad quality, worst quality, deformed, extra limbs...", 
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": cfg_val,  # Der variable Schleifenwert der Testreihe
        "cfg_normalization": True,
        "width": img_cfg.get("width", 1024),
        "height": img_cfg.get("height", 1024),
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "note": "Z-Image 6B Eastern Model CFG Ablation Test in native bf16"
    }

    with open(META_DIR / f"{image_id}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

# =============================================================
# MODELL LADEN (6B NATIV AUF GPU)
# =============================================================
def load_model(model_cfg, logger):
    model_id = model_cfg.get("models", {}).get("zimage", {}).get("model_id", "Tongyi-MAI/Z-Image")
    dtype = torch.bfloat16

    logger.info(f"Lade östliches Z-Image Modell für CFG-Test: {model_id}")
    
    pipe = ZImagePipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    pipe.to("cuda")
    
    # Anti-Freeze-Optimierungen für stabilen Batch-Betrieb
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"Z-Image erfolgreich geladen! VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return pipe

# =============================================================
# GENERIERUNG
# =============================================================
def generate_image(pipe, prompt_info, seed, cfg_val, model_cfg, logger):
    zimage_cfg = model_cfg.get("models", {}).get("zimage", {}).get("generation", {})
    steps = zimage_cfg.get("num_inference_steps", 50)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    TECHNICAL_NEGATIVE_PROMPT = "bad quality, worst quality, deformed, extra limbs, floating objects, surreal, abstract, artifacts, messy background"

    image = pipe(
        prompt=prompt_info["prompt"],
        negative_prompt=TECHNICAL_NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=cfg_val,       # Dynamische Injektion aus der Testschleife
        cfg_normalization=True,       # Architektonisch zwingend erforderlich bei Z-Image
        generator=generator,
    ).images[0]

    gen_time = time.time() - start
    return image, gen_time, steps

# =============================================================
# MAIN PIPELINE EXECUTION
# =============================================================
def main(dry_run=False, resume=True):
    logger = setup_logging()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTE Z-IMAGE CFG-ABLATION-TEST")
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
        logger.info(f"Checkpoint geladen: {len(completed)} Bilder bereits fertig.")

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
                    image, gen_time, steps = generate_image(pipe, prompt_info, seed, cfg_val, model_cfg, logger)

                    image_path = IMAGE_DIR / f"{image_id}.png"
                    image.save(image_path, format="PNG")

                    save_metadata(image_id, prompt_info, seed, cfg_val, model_cfg, gen_time, image_path, steps)

                    completed.add(image_id)
                    save_checkpoint(completed)
                    success_count += 1

                    elapsed = time.time() - total_start
                    eta = (elapsed / success_count) * (total_images - success_count) if success_count > 0 else 0
                    logger.info(f"   {gen_time:.2f}s | Fortschritt: {success_count}/{total_images} | ETA: {eta/60:.1f} min")

                except Exception as e:
                    logger.error(f"   Fehler bei {image_id}: {e}")
                    failed.append({"id": image_id, "error": str(e)})
                    torch.cuda.empty_cache()
                    gc.collect()

    total_time = time.time() - total_start
    logger.info(f"\n🎉 Z-IMAGE CFG-TEST BEENDET! Erfolgreich: {len(completed)}/{total_images} | Fehler: {len(failed)}")

    if failed:
        with open(OUTPUT_DIR / "failed_zimage_cfg_test.json", "w") as f:
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