"""
pipeline.py
===========
Master-Runner - Startet alle Modelle nacheinander (inkl. FLUX.2-klein-9B)

Lädt jedes Modell, generiert alle Bilder, löscht VRAM, nächstes Modell.
So braucht es nie mehr als den VRAM des aktuell laufenden Modells.

Verwendung:
    python pipeline.py                    # Alle 4 Modelle
    python pipeline.py --model sd35       # Nur SD 3.5
    python pipeline.py --model flux       # Nur Flux.2-dev (Das Große)
    python pipeline.py --model flux_klein # Nur Flux.2-klein-9B (Das Schnelle)
    python pipeline.py --model qwen       # Nur Qwen
    python pipeline.py --dry-run          # Test ohne Generierung
"""

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import generation.run_sd35 as sd35_runner
import generation.run_flux as flux_runner
import generation.run_flux_9b as flux_klein_runner 
import generation.run_qwen as qwen_runner
import generation.run_zimage as zimage_runner


def main():
    parser = argparse.ArgumentParser(description="Bias Evaluation - Master Pipeline")
    parser.add_argument("--model", choices=["sd35", "flux", "flux_klein", "qwen", "all"],
                        default="all", help="Welches Modell ausführen")
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
    logger = logging.getLogger("pipeline")

    logger.info("=" * 60)
    logger.info("BIAS EVALUATION PIPELINE - Master Runner")
    logger.info(f"Gestartet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Modus: {'DRY-RUN' if args.dry_run else 'PRODUCTION'}")
    logger.info("=" * 60)

    resume = not args.no_resume
    dry_run = args.dry_run
    model = args.model

    if model in ("sd35", "all"):
        logger.info("\n" + "🔵 " * 20)
        logger.info("STARTE: Stable Diffusion 3.5 Large")
        logger.info("🔵 " * 20)
        sd35_runner.main(dry_run=dry_run, resume=resume)

    # if model in ("flux", "all"):
    #     logger.info("\n" + "🟡 " * 20)
    #     logger.info("STARTE: FLUX.2-dev (32B)")
    #     logger.info("🟡 " * 20)
    #     flux_runner.main(dry_run=dry_run, resume=resume)

    # --- NEU: Der Block für das 9B Modell ---
    # if model in ("flux_klein", "all"):
    #     logger.info("\n" + "🟠 " * 20)
    #     logger.info("STARTE: FLUX.2-klein-9B")
    #     logger.info("🟠 " * 20)
    #     flux_klein_runner.main(dry_run=dry_run, resume=resume)

    # if model in ("qwen", "all"):
    #     logger.info("\n" + "🟢 " * 20)
    #     logger.info("STARTE: Qwen-Image-2512")
    #     logger.info("🟢 " * 20)
    #     qwen_runner.main(dry_run=dry_run, resume=resume)

    # if model in ("zimage", "all"):
    #     logger.info("\n" + "� " * 20)
    #     logger.info("STARTE: Z-Image")
    #     logger.info("� " * 20)
    #     zimage_runner.main(dry_run=dry_run, resume=resume)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE KOMPLETT ABGESCHLOSSEN")
    logger.info(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Nächster Schritt: python analysis/detect_persons.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()