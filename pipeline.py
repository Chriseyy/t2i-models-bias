"""
pipeline.py
===========
Master-Runner - Startet alle Modelle nacheinander (inkl. FLUX.2-klein-9B)

Lädt jedes Modell, generiert alle Bilder, löscht VRAM, nächstes Modell.
So braucht es nie mehr als den VRAM des aktuell laufenden Modells.

Verwendung:
    python pipeline.py                    # Alle Modelle (Englisch)
    python pipeline.py --chinese          # Alle Modelle (Chinesisch Deep Dive)
    python pipeline.py --model flux_klein # Nur Flux.2-klein-9B (Englisch)
    python pipeline.py --model zimage --chinese # Nur Z-Image (Chinesisch)
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
    # "zimage" zu den choices hinzugefügt, damit das Argument fehlerfrei akzeptiert wird
    parser.add_argument("--model", choices=["sd35", "flux", "flux_klein", "qwen", "zimage", "all"],
                        default="all", help="Welches Modell ausführen")
    parser.add_argument("--dry-run", action="store_true",
                        help="Nur testen ohne echte Generierung")
    parser.add_argument("--no-resume", action="store_true",
                        help="Checkpoint ignorieren")
    # Das neue Flag für den Cross-Lingual Support
    parser.add_argument("--chinese", action="store_true",
                        help="Startet den chinesischen Cross-Lingual Deep Dive für alle ausgewählten Modelle")
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
    logger.info(f"Modus:    {'DRY-RUN' if args.dry_run else 'PRODUCTION'}")
    logger.info(f"Sprache:  {'CHINESISCH (Deep Dive)' if args.chinese else 'ENGLISCH (Main)'}")
    logger.info("=" * 60)

    resume = not args.no_resume
    dry_run = args.dry_run
    model = args.model
    chinese = args.chinese # Reicht den Parameter nach unten weiter

    if model in ("sd35", "all"):
        logger.info("\n" + "🔵 " * 20)
        logger.info("STARTE: Stable Diffusion 3.5 Large")
        logger.info("🔵 " * 20)
        sd35_runner.main(dry_run=dry_run, resume=resume, chinese=chinese)

    if model in ("flux", "all"):
        logger.info("\n" + "🟡 " * 20)
        logger.info("STARTE: FLUX.2-dev (32B)")
        logger.info("🟡 " * 20)
        flux_runner.main(dry_run=dry_run, resume=resume, chinese=chinese)

    if model in ("flux_klein", "all"):
        logger.info("\n" + "export 🟠 " * 20)
        logger.info("STARTE: FLUX.2-klein-9B")
        logger.info("🟠 " * 20)
        flux_klein_runner.main(dry_run=dry_run, resume=resume, chinese=chinese)

    if model in ("qwen", "all"):
        logger.info("\n" + "🟢 " * 20)
        logger.info("STARTE: Qwen-Image-2512")
        logger.info("🟢 " * 20)
        qwen_runner.main(dry_run=dry_run, resume=resume, chinese=chinese)

    if model in ("zimage", "all"):
        logger.info("\n" + "🔮 " * 20)
        logger.info("STARTE: Z-Image (6B)")
        logger.info("🔮 " * 20)
        zimage_runner.main(dry_run=dry_run, resume=resume, chinese=chinese)

    logger.info("\n" + "=" * 60)
    logger.info("✅ PIPELINE KOMPLETT ABGESCHLOSSEN")
    logger.info(f"Beendet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Nächster Schritt: python analysis/detect_persons.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()