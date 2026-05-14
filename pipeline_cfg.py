"""
pipeline_cfg.py
===============
Master-Runner für die CFG-Evaluation (Guidance Scale Tests).
Führt die speziellen CFG-Testskripte nacheinander aus.

Verwendung:
    python pipeline_cfg.py                    # Alle CFG-Tests
    python pipeline_cfg.py --model flux_klein # Nur CFG-Test für Flux.2-klein-9B
    python pipeline_cfg.py --dry-run          # Test ohne Generierung
"""

import argparse
import logging
import sys
from datetime import datetime

# Fügt den aktuellen Ordner zum Python-Pfad hinzu
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

# Importiere hier deine neuen CFG-Test-Skripte
import generation.run_flux_cfg_test as flux_klein_cfg_runner 


def main():
    parser = argparse.ArgumentParser(description="Bias Evaluation - CFG Master Pipeline")
    
    # Füge hier später andere Modelle hinzu, falls du auch für andere Modelle CFG-Tests baust
    parser.add_argument("--model", choices=["flux_klein", "all"],
                        default="all", help="Welches CFG-Test-Modell ausführen")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur testen ohne echte Generierung")
    parser.add_argument("--no-resume", action="store_true",
                        help="Checkpoint ignorieren")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout
    )
    logger = logging.getLogger("pipeline_cfg")

    logger.info("=" * 60)
    logger.info("CFG EVALUATION PIPELINE - Master Runner")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modus: {'DRY-RUN' if args.dry_run else 'PRODUCTION'}")
    logger.info("=" * 60)

    resume = not args.no_resume
    dry_run = args.dry_run
    model = args.model

    if model in ("flux_klein", "all"):
        logger.info("\n" + "🟠 " * 20)
        logger.info("STARTE CFG-TEST: FLUX.2-klein-9B")
        logger.info("🟠 " * 20)
        flux_klein_cfg_runner.main(dry_run=dry_run, resume=resume)

    # Platzhalter für zukünftige Tests (z.B. Z-Image CFG Test)
    # if model in ("zimage", "all"):
    #     logger.info("\n" + "🔵 " * 20)
    #     logger.info("STARTE CFG-TEST: Z-Image")
    #     logger.info("🔵 " * 20)
    #     zimage_cfg_runner.main(dry_run=dry_run, resume=resume)

    logger.info("\n" + "=" * 60)
    logger.info("✅ CFG-PIPELINE KOMPLETT ABGESCHLOSSEN")
    logger.info(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Die Bilder liegen nun in 'outputs/images_cfg_test'")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()