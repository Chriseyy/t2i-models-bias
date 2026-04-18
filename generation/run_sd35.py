"""
run_sd35.py
===========
Generation-Script für Stable Diffusion 3.5 Large

Features:
- Checkpoint-System: Startet da weiter wo es aufgehört hat
- Metadata-JSON pro Bild (für spätere Analyse)
- Sauberes Error-Logging (kein stiller Fehler!)
- VRAM-Cleanup nach jedem Batch

Verwendung:
    python generation/run_sd35.py
    python generation/run_sd35.py --dry-run    # Nur testen ohne zu generieren
    python generation/run_sd35.py --resume     # Checkpoint weitermachen
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

# Lädt Umgebungsvariablen aus .env (z.B. HuggingFace Token)
# =============================================================
# PFADE - Relativ zum Projekt-Root
# =============================================================
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
IMAGE_DIR    = OUTPUT_DIR / "images" / "sd35"
META_DIR     = OUTPUT_DIR / "metadata" / "sd35"
LOG_DIR      = OUTPUT_DIR
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint_sd35.json"
env_path = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=env_path)

# =============================================================
# LOGGING SETUP
# =============================================================
def setup_logging():
    log_file = LOG_DIR / "generation_sd35.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("sd35")


# =============================================================
# CONFIG LADEN
# =============================================================
def load_configs():
    """Lädt prompts.yaml und models.yaml mit Sicherheits-Check"""
    prompt_path = CONFIG_DIR / "prompts.yaml"
    model_path = CONFIG_DIR / "models.yaml"

    print(f"\n🔍 Lese Prompts von: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)

    print(f"🔍 Lese Models von: {model_path}")
    with open(model_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Sicherheits-Checks
    if prompt_cfg is None:
        raise ValueError(f"❌ FEHLER: Die Datei {prompt_path} ist leer!")
    if model_cfg is None:
        raise ValueError(f"❌ FEHLER: Die Datei {model_path} ist leer! (Bitte speichern oder Pfad prüfen)")

    return prompt_cfg, model_cfg


# =============================================================
# ALLE PROMPTS ALS FLACHE LISTE AUFBAUEN
# =============================================================
def build_prompt_list(prompt_cfg):
    """
    Baut aus der YAML-Struktur eine flache Liste:
    [{"id": "prof_doctor", "subject": "a doctor", "category": "professions_high_status", ...}, ...]
    """
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
                "expected_bias": item.get("expected_bias", "unknown"),
                "category_description": category_data.get("description", "")
            })

    return all_prompts


# =============================================================
# CHECKPOINT SYSTEM
# =============================================================
def load_checkpoint():
    """Lädt den Checkpoint - welche Bilder wurden schon generiert?"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_checkpoint(completed_ids: set):
    """Speichert welche Image-IDs schon fertig sind"""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "completed": list(completed_ids),
            "last_updated": datetime.now().isoformat(),
            "model": "sd35"
        }, f, indent=2)


def make_image_id(prompt_id: str, seed: int) -> str:
    """Eindeutige ID pro Bild: z.B. 'prof_doctor_seed042'"""
    return f"{prompt_id}_seed{seed:03d}"


# =============================================================
# METADATA SPEICHERN
# =============================================================
def save_metadata(image_id: str, prompt_info: dict, seed: int,
                  model_cfg: dict, generation_time: float,
                  image_path: str):
    """
    Speichert JSON-Metadata für jedes Bild.
    Wird später von der Analyse-Pipeline genutzt!
    """
    meta = {
        "image_id": image_id,
        "model": "sd35",
        "model_full_name": "Stable Diffusion 3.5 Large",
        "model_id": model_cfg["models"]["sd35"]["model_id"],

        # Prompt-Info
        "prompt_id": prompt_info["id"],
        "prompt": prompt_info["prompt"],
        "subject": prompt_info["subject"],
        "category": prompt_info["category"],
        "expected_bias": prompt_info["expected_bias"],

        # Generation-Parameter
        "seed": seed,
        "num_inference_steps": model_cfg["models"]["sd35"]["generation"]["num_inference_steps"],
        "guidance_scale": model_cfg["models"]["sd35"]["generation"]["guidance_scale"],
        "width": model_cfg["global"]["output_size"]["width"],
        "height": model_cfg["global"]["output_size"]["height"],
        "dtype": model_cfg["global"]["dtype"],

        # Ergebnis
        "image_path": str(image_path),
        "generation_time_seconds": round(generation_time, 2),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    }

    meta_path = META_DIR / f"{image_id}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta_path


def load_model(model_cfg: dict, logger: logging.Logger):
    """
    Lädt SD 3.5 Pipeline.
    WICHTIG: Zuerst HuggingFace Login prüfen!
    """
    model_id = model_cfg["models"]["sd35"]["model_id"]
    dtype_str = model_cfg["global"]["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    logger.info(f"Lade Modell: {model_id}")
    logger.info(f"dtype={dtype_str} | device=cuda")
    logger.info("Dies kann beim ersten Mal einige Minuten dauern (Download ~12GB)...")

    # Token aus der .env Datei holen
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        logger.error("❌ Kein HF_TOKEN gefunden! Bitte .env Datei prüfen.")
        raise ValueError("Missing Hugging Face Token")


    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token
        )
        
        pipe = pipe.to("cuda")
        # pipe.enable_model_cpu_offload() 

        # torch.backends.cuda.matmul.allow_tf32 = True
        # pipe.transformer.to(memory_format=torch.channels_last)
        # pipe.vae.to(memory_format=torch.channels_last)
        
        logger.info("✅ Modell geladen und optimiert!")
        logger.info(f"   VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        return pipe

    except Exception as e:
        logger.error(f"❌ Modell konnte nicht geladen werden: {e}")
        logger.error("   Prüfe: 1) huggingface-cli login  2) Lizenz auf HF akzeptiert?")
        raise


# =============================================================
# BILD GENERIEREN
# =============================================================
def generate_image(pipe, prompt_info: dict, seed: int,
                   model_cfg: dict, logger: logging.Logger):
    """
    Generiert ein einzelnes Bild und gibt (image, generation_time) zurück.
    """
    gen_cfg = model_cfg["models"]["sd35"]["generation"]
    img_cfg = model_cfg["global"]["output_size"]
    neg_prompt = model_cfg["prompts"]["negative_prompt"] if "prompts" in model_cfg else \
                 "cartoon, anime, illustration, painting, drawing, unrealistic"

    generator = torch.Generator(device="cuda").manual_seed(seed)

    start_time = time.time()

    image = pipe(
        prompt=prompt_info["prompt"],
        negative_prompt=neg_prompt if gen_cfg["use_negative_prompt"] else None,
        width=img_cfg["width"],
        height=img_cfg["height"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
        generator=generator,
        num_images_per_prompt=1,
    ).images[0]

    generation_time = time.time() - start_time

    return image, generation_time


# =============================================================
# HAUPTPROGRAMM
# =============================================================
def main(dry_run: bool = False, resume: bool = True):
    # Setup
    logger = setup_logging()
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BIAS EVALUATION - SD 3.5 Generation")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Config laden
    prompt_cfg, model_cfg = load_configs()
    prompts = build_prompt_list(prompt_cfg)
    seeds   = prompt_cfg["seeds"]

    # Stats berechnen
    total_images = len(prompts) * len(seeds)
    logger.info(f"Prompts: {len(prompts)} | Seeds: {len(seeds)} | Gesamt: {total_images} Bilder")

    # Checkpoint laden
    completed = load_checkpoint() if resume else set()
    if completed:
        logger.info(f"Checkpoint gefunden: {len(completed)} Bilder bereits fertig → werden übersprungen")

    # Dry-Run: Nur zeigen was generiert werden würde
    if dry_run:
        logger.info("\n🔍 DRY-RUN - Keine echten Bilder werden generiert!")
        for p in prompts[:3]:
            for s in seeds[:2]:
                img_id = make_image_id(p["id"], s)
                status = "✅ DONE" if img_id in completed else "⏳ TODO"
                logger.info(f"  {status} | {img_id} | {p['prompt'][:60]}...")
        logger.info("...")
        logger.info(f"\nGesamt: {total_images} Bilder | Fertig: {len(completed)} | Offen: {total_images - len(completed)}")
        return

    # Modell laden
    logger.info("\nLade Modell...")
    pipe = load_model(model_cfg, logger)

    # ==========================================================
    # GENERATION LOOP
    # ==========================================================
    failed = []
    success_count = len(completed)
    total_start = time.time()

    for prompt_info in prompts:
        for seed in seeds:
            image_id = make_image_id(prompt_info["id"], seed)

            # Schon generiert? Überspringen!
            if image_id in completed:
                logger.info(f"  ⏭️  Überspringe (bereits fertig): {image_id}")
                continue

            logger.info(f"\n📸 Generiere: {image_id}")
            logger.info(f"   Prompt: {prompt_info['prompt'][:80]}...")
            logger.info(f"   Seed: {seed} | Steps: {model_cfg['models']['sd35']['generation']['num_inference_steps']}")

            try:
                # Bild generieren
                image, gen_time = generate_image(pipe, prompt_info, seed, model_cfg, logger)

                # Bild speichern
                image_path = IMAGE_DIR / f"{image_id}.png"
                image.save(image_path, format="PNG")

                # Metadata speichern
                save_metadata(
                    image_id=image_id,
                    prompt_info=prompt_info,
                    seed=seed,
                    model_cfg=model_cfg,
                    generation_time=gen_time,
                    image_path=image_path
                )

                # Checkpoint updaten
                completed.add(image_id)
                save_checkpoint(completed)
                success_count += 1

                logger.info(f"   ✅ Fertig in {gen_time:.1f}s → {image_path.name}")

                # Fortschritt anzeigen
                progress = success_count / total_images * 100
                elapsed = time.time() - total_start
                eta = (elapsed / success_count) * (total_images - success_count) if success_count > 0 else 0
                logger.info(f"   📊 Fortschritt: {success_count}/{total_images} ({progress:.1f}%) | ETA: {eta/60:.1f} min")

            except torch.cuda.OutOfMemoryError:
                logger.error(f"   ❌ GPU OOM bei {image_id}! Versuche VRAM zu leeren...")
                torch.cuda.empty_cache()
                gc.collect()
                failed.append({"id": image_id, "error": "CUDA_OOM"})

            except Exception as e:
                logger.error(f"   ❌ Fehler bei {image_id}: {e}")
                failed.append({"id": image_id, "error": str(e)})

    # ==========================================================
    # ABSCHLUSS-REPORT
    # ==========================================================
    total_time = time.time() - total_start

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION ABGESCHLOSSEN")
    logger.info("=" * 60)
    logger.info(f"✅ Erfolgreich: {len(completed)}/{total_images}")
    logger.info(f"❌ Fehlgeschlagen: {len(failed)}")
    logger.info(f"⏱️  Gesamtzeit: {total_time/60:.1f} Minuten")
    logger.info(f"📁 Bilder in: {IMAGE_DIR}")
    logger.info(f"📋 Metadata in: {META_DIR}")

    # Fehlgeschlagene Bilder speichern
    if failed:
        failed_path = OUTPUT_DIR / "failed_sd35.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        logger.info(f"⚠️  Fehler-Log: {failed_path}")

    # VRAM freigeben
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("🧹 VRAM geleert")


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SD 3.5 Bias Evaluation Generator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur zeigen was generiert werden würde, ohne echte Bilder")
    parser.add_argument("--no-resume", action="store_true",
                        help="Checkpoint ignorieren, alles neu generieren")
    args = parser.parse_args()

    main(
        dry_run=args.dry_run,
        resume=not args.no_resume
    )