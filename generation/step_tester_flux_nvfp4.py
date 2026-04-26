"""
step_tester_flux.py
===================
Ablation Study für FLUX.2-dev-NVFP4
Zeigt, wie das Modell in gewaltigen Sprüngen (1 bis 4 Steps) 
das Bild formt, im Gegensatz zu klassischen Diffusionsmodellen.
"""

import os
import torch
import time
from pathlib import Path
from diffusers import Flux2Pipeline  # Geändert auf die Standard FluxPipeline
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/flux_nvfp4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter
PROMPT = "a photo of a beautiful person"
SEED = 103
# Zu testende Steps
STEPS_TO_TEST = [1, 3, 5, 10, 20, 30, 40, 45, 50, 55, 60]

def main():
    print("=" * 60)
    print("🔬 KI-ZEITLUPE: FLUX.2-dev-NVFP4")
    print("=" * 60)
    
    # 3. Modell aus dem Hub in den Cache laden (oder Pfad abfragen, falls schon da)
    print("Prüfe/Lade Checkpoint aus dem Hugging Face Hub (kann beim ersten Mal dauern)...")
    try:
        ckpt_path = hf_hub_download(
            repo_id="black-forest-labs/FLUX.2-dev-NVFP4",
            # filename="flux2-dev-nvfp4.safetensors",
            filename="flux2-dev-nvfp4-mixed.safetensors",
            token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
        )
        print(f"Checkpoint gefunden unter: {ckpt_path}")
    except Exception as e:
        print(f"Fehler beim Download: {e}")
        return

    # 4. Modell laden
    print("Lade FLUX.2-dev-NVFP4 über from_single_file...")
    pipe = Flux2Pipeline.from_single_file(
        ckpt_path, # <--- HIER übergeben wir nun den sauberen lokalen Pfad
        torch_dtype=torch.bfloat16
    )
    
    pipe.enable_model_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    print("\n🚀 Starte Generierungen...")
    
    # 5. Loop durch die Steps
    for steps in STEPS_TO_TEST:
        # Generator bei JEDEM Step neu mit Seed starten
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=steps,
            guidance_scale=4.0,  # FLUX.dev Modelle laufen typischerweise am besten bei ~3.5
            generator=generator,
        ).images[0]
        gen_time = time.time() - start_time
        
        # 5. Speichern
        filename = OUTPUT_DIR / f"doctor_seed{SEED}_{steps:02d}steps.png"
        image.save(filename)
        
        print(f"✅ Bild mit {steps:02d} Steps gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
        
    print("\n🎉 FLUX Testbilder generiert! Schau in den Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()