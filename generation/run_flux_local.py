"""
run_flux.py
===========
Generation-Script für FLUX.2-dev (4-Bit & Remote Text Encoder)

Wichtiger Unterschied zu SD 3.5:
- Nutzt einen Cloud-Server für die Text-Vektoren (spart ~20GB VRAM!)
- Nutzt das 4-Bit quantisierte Modell für maximale RTX 5090 Performance
- Keine negativen Prompts

Verwendung:
    python generation/run_flux.py
"""

import os
import sys
import json
import time
import logging
import argparse
import gc
import io
import requests
from datetime import datetime
from pathlib import Path

import yaml
import torch
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image
from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import get_token

# =============================================================
# PFADE
# =============================================================
PROJECT_ROOT    = Path(__file__).parent.parent
CONFIG_DIR      = PROJECT_ROOT / "config"
OUTPUT_DIR      = PROJECT_ROOT / "outputs"
IMAGE_DIR       = OUTPUT_DIR / "images" / "flux"
META_DIR        = OUTPUT_DIR / "metadata" / "flux"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_flux.json"

env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# LOGGING
# =============================================================
def setup_logging():
    log_file = OUTPUT_DIR / "generation_flux.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("flux")

# =============================================================
# CONFIG
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
            full_prompt = base_template.format(subject=item["subject"])
            all_prompts.append({
                "id": item["id"],
                "subject": item["subject"],
                "prompt": full_prompt,
                "category": category_name,
                "expected_bias": item.get("expected_bias", "unknown")
            })
    return all_prompts

# =============================================================
# CHECKPOINT & METADATA
# =============================================================
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
            "model": "flux"
        }, f, indent=2)

def make_image_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_seed{seed:03d}"

def save_metadata(image_id, prompt_info, seed, model_cfg, gen_time, image_path):
    gen_cfg = model_cfg["models"]["flux"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]

    meta = {
        "image_id": image_id,
        "model": "flux",
        "model_full_name": model_cfg["models"]["flux"]["name"],
        "model_id": model_cfg["models"]["flux"]["model_id"],
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],
        "seed": seed,
        "num_inference_steps": gen_cfg["num_inference_steps"],
        "guidance_scale": gen_cfg["guidance_scale"],
        "width": img_cfg["width"],
        "height": img_cfg["height"],
        "generation_time_seconds": round(gen_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "note": "Used remote text encoder and 4-bit quantization"
    }

    meta_path = META_DIR / f"{image_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta_path

# =============================================================
# REMOTE TEXT ENCODER API
# =============================================================
def get_remote_prompt_embeds(prompt: str, logger: logging.Logger):
    """Holt die Text-Vektoren vom HuggingFace Server, statt sie lokal zu berechnen."""
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or get_token()
    if not token:
        raise ValueError("Kein HuggingFace Token gefunden!")

    try:
        response = requests.post(
            "https://remote-text-encoder-flux-2.huggingface.co/predict",
            json={"prompt": prompt},
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            timeout=45 # Timeout, falls der Server hängt
        )
        response.raise_for_status() # Löst Fehler aus, wenn Statuscode != 200
        # prompt_embeds = torch.load(io.BytesIO(response.content))
        prompt_embeds = torch.load(io.BytesIO(response.content), weights_only=False)
        return prompt_embeds.to("cuda")
    except Exception as e:
        logger.error(f"❌ Fehler bei der Server-Kommunikation für den Text-Encoder: {e}")
        raise

# =============================================================
# MODELL LADEN
# =============================================================
def load_model(model_cfg, logger):
    model_id = model_cfg["models"]["flux"]["model_id"]
    dtype = torch.bfloat16

    logger.info(f"Lade 100% lokales 4-Bit Modell: {model_id}")
    
    # 1. Text-Encoder manuell laden (Umgeht den Pixtral-Bug!)
    logger.info("   Lade 4-Bit Text-Encoder...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
    )
    
    # 2. Transformer (DiT) manuell laden
    logger.info("   Lade 4-Bit Transformer...")
    dit = AutoModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
    )
    
    # 3. Pipeline zusammenbauen
    logger.info("   Baue Pipeline zusammen...")
    pipe = Flux2Pipeline.from_pretrained(
        model_id, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype
    )
    
    # Der magische Offload (Staffellauf zwischen RAM und VRAM)
    pipe.enable_model_cpu_offload()

    # 5090 Optimierungen
    torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("✅ FLUX erfolgreich geladen!")
    return pipe

# =============================================================
# GENERIERUNG
# =============================================================
def generate_image(pipe, prompt_info, seed, model_cfg, logger):
    gen_cfg = model_cfg["models"]["flux"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]

    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.time()

    logger.info("   🖌️ Generiere Bild lokal...")
    image = pipe(
        prompt=prompt_info["prompt"], 
        width=img_cfg["width"],
        height=img_cfg["height"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
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
    logger.info("BIAS EVALUATION - FLUX.2-dev (4-Bit Remote)")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    prompt_cfg, model_cfg = load_configs()
    prompts = build_prompt_list(prompt_cfg)
    seeds   = prompt_cfg["seeds"]
    total_images = len(prompts) * len(seeds)

    logger.info(f"Prompts: {len(prompts)} | Seeds: {len(seeds)} | Gesamt: {total_images}")

    completed = load_checkpoint() if resume else set()
    if completed:
        logger.info(f"Checkpoint: {len(completed)} bereits fertig")

    if dry_run:
        logger.info("\n🔍 DRY-RUN")
        for p in prompts[:3]:
            for s in seeds[:2]:
                img_id = make_image_id(p["id"], s)
                status = "✅" if img_id in completed else "⏳"
                logger.info(f"  {status} {img_id}")
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
            logger.info(f"   {prompt_info['prompt'][:80]}...")

            try:
                image, gen_time = generate_image(pipe, prompt_info, seed, model_cfg, logger)

                image_path = IMAGE_DIR / f"{image_id}.png"
                image.save(image_path, format="PNG")

                save_metadata(image_id, prompt_info, seed, model_cfg, gen_time, image_path)

                completed.add(image_id)
                save_checkpoint(completed)
                success_count += 1

                elapsed = time.time() - total_start
                eta = (elapsed / success_count) * (total_images - success_count) if success_count > 0 else 0
                logger.info(f"   ✅ {gen_time:.1f}s | {success_count}/{total_images} | ETA: {eta/60:.1f} min")

            except Exception as e:
                # Fängt Fehler beim API-Call UND bei CUDA ab
                logger.error(f"   ❌ {image_id}: {e}")
                failed.append({"id": image_id, "error": str(e)})
                
                # Sicherheitsnetz: VRAM leeren bei Fehlern
                torch.cuda.empty_cache()
                gc.collect()

    # Abschluss
    total_time = time.time() - total_start
    logger.info(f"\n✅ {len(completed)}/{total_images} | ❌ {len(failed)} | ⏱️ {total_time/60:.1f} min")

    if failed:
        with open(OUTPUT_DIR / "failed_flux.json", "w") as f:
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