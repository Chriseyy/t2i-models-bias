"""
pipeline.py
===========
Master-Runner - Startet alle 3 Modelle nacheinander

Lädt jedes Modell, generiert alle Bilder, löscht VRAM, nächstes Modell.
So braucht es nie mehr als ~16GB VRAM gleichzeitig.

Verwendung:
    python pipeline.py                    # Alle 3 Modelle
    python pipeline.py --model sd35       # Nur SD 3.5
    python pipeline.py --model flux       # Nur Flux
    python pipeline.py --model qwen       # Nur Qwen
    python pipeline.py --dry-run          # Test ohne Generierung
"""

import argparse
import logging
import sys
from datetime import datetime

sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import generation.run_sd35 as sd35_runner
# import generation.run_flux as flux_runner
# import generation.run_qwen as qwen_runner


def main():
    parser = argparse.ArgumentParser(description="Bias Evaluation - Master Pipeline")
    parser.add_argument("--model", choices=["sd35", "flux", "qwen", "all"],
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
    #     logger.info("STARTE: FLUX.2-dev")
    #     logger.info("🟡 " * 20)
    #     flux_runner.main(dry_run=dry_run, resume=resume)

    # if model in ("qwen", "all"):
    #     logger.info("\n" + "🟢 " * 20)
    #     logger.info("STARTE: Qwen-Image-2512")
    #     logger.info("🟢 " * 20)
    #     qwen_runner.main(dry_run=dry_run, resume=resume)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE KOMPLETT ABGESCHLOSSEN")
    logger.info(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Nächster Schritt: python analysis/detect_persons.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()