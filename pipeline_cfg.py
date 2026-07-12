"""
pipeline_cfg.py
===============
Master-Runner für die CFG-Evaluation (Guidance Scale Tests).
Führt die speziellen CFG-Testskripte nacheinander aus.

Verwendung:
    python pipeline_cfg.py                    # Alle CFG-Tests nacheinander
    python pipeline_cfg.py --model flux_klein # Nur CFG-Test für Flux.2-klein-9B
    python pipeline_cfg.py --model zimage     # Nur CFG-Test für Z-Image
    python pipeline_cfg.py --dry-run          # Test ohne Generierung
"""

import argparse
import logging
import sys
from datetime import datetime

# Fügt den aktuellen Ordner zum Python-Pfad hinzu
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

# Importiere alle spezifischen CFG-Test-Skripte aus dem generation-Ordner
import generation.run_flux_9b_cfg_test as flux_klein_cfg_runner 
import generation.run_flux_cfg_test as flux_dev_cfg_runner
import generation.run_qwen_cfg_test as qwen_cfg_runner
import generation.run_zimage_cfg_test as zimage_cfg_runner
import generation.run_sd35_cfg_test as sd35_cfg_runner


def main():
    parser = argparse.ArgumentParser(description="Bias Evaluation - CFG Master Pipeline")
    
    # Alle Modelle in den Choices hinzugefügt
    parser.add_argument("--model", choices=["flux_klein", "flux_dev", "qwen", "zimage", "sd35", "all"],
                        default="all", help="Welches CFG-Test-Modell ausgeführt werden soll")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur testen ohne echte Generierung")
    parser.add_argument("--no-resume", action="store_true",
                        help="Checkpoint ignorieren und bei 0 starten")
    args = parser.parse_args()

    # Basis-Logging Setup für den Master-Runner
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout
    )
    logger = logging.getLogger("pipeline_cfg")

    logger.info("=" * 60)
    logger.info("CFG EVALUATION PIPELINE - Master Runner")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modus:   {'DRY-RUN' if args.dry_run else 'PRODUCTION'}")
    logger.info(f"Modell:  {args.model.upper()}")
    logger.info("=" * 60)

    resume = not args.no_resume
    dry_run = args.dry_run
    model = args.model

    # 1. FLUX.2-klein-9B (FP16)
    if model in ("flux_klein", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STARTE CFG-TEST: FLUX.2-klein-9B (FP16)")
        logger.info("=" * 60)
        flux_klein_cfg_runner.main(dry_run=dry_run, resume=resume)

    # 2. FLUX.2-dev (4-Bit)
    if model in ("flux_dev", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STARTE CFG-TEST: FLUX.2-dev (4-Bit Quantized)")
        logger.info("=" * 60)
        flux_dev_cfg_runner.main(dry_run=dry_run, resume=resume)

    # 3. Stable Diffusion 3.5 Large (FP16)
    if model in ("sd35", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STARTE CFG-TEST: Stable Diffusion 3.5 Large (FP16)")
        logger.info("=" * 60)
        sd35_cfg_runner.main(dry_run=dry_run, resume=resume)

    # 4. Qwen-Image (4-Bit)
    if model in ("qwen", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STARTE CFG-TEST: Qwen-Image (4-Bit Unsloth)")
        logger.info("=" * 60)
        qwen_cfg_runner.main(dry_run=dry_run, resume=resume)

    # 5. Z-Image (FP16)
    if model in ("zimage", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("STARTE CFG-TEST: Z-Image (FP16)")
        logger.info("=" * 60)
        zimage_cfg_runner.main(dry_run=dry_run, resume=resume)

    logger.info("\n" + "=" * 60)
    logger.info("✅ CFG-PIPELINE KOMPLETT ABGESCHLOSSEN")
    logger.info(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Die generierten Bilder liegen in 'outputs/images_cfg_test'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()