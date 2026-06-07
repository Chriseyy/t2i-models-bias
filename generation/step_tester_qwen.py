"""
step_tester_qwen.py
===================
Ablation Study für Qwen-Image-2512 (Unsloth 4-Bit).
Untersucht die Trajektorienstabilität des chinesischen Modells bei 
unterschiedlichen Inferenzschritten (1 bis 60 Steps).
"""

import os
import torch
import time
import gc
from pathlib import Path
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/qwen")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter (Analog zu deinem SD3.5/Flux-Szenario)
PROMPT = "a photo of a doctor"
SEED = 103

STEPS_TO_TEST = [1, 3, 5, 10, 20, 30, 40, 45, 50, 55, 60]

def main():
    print("=" * 60)
    print("🔬 KI-ZEITLUPE: QWEN-IMAGE-2512 (Unsloth 4-Bit)")
    print("=" * 60)
    
    # 3. Qwen Modell via Unsloth 4-Bit laden
    model_id = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
    print(f"Lade Qwen-Modell in 4-Bit: {model_id}...")
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    print(f"✅ Qwen erfolgreich im VRAM verankert: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print("\n🚀 Starte Generierungs-Schleife...")
    
    # 4. Loop durch die Sampling Steps
    for steps in STEPS_TO_TEST:
        # Extrem wichtig: Generator vor JEDEM Durchlauf exakt mit dem Seed resetten!
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        try:
            # Qwen Inferenz aufrufen
            output = pipe(
                prompt=PROMPT,
                num_inference_steps=steps,
                true_cfg_scale=4.0,  # Qwen nutzt true_cfg_scale statt guidance_scale
                generator=generator,
            )
            image = output.images[0]
            gen_time = time.time() - start_time
            
            # 5. Bild abspeichern
            filename = OUTPUT_DIR / f"doctor_seed{SEED}_{steps:02d}steps.png"
            image.save(filename, format="PNG")
            
            print(f"✅ [{steps:02d} Steps] gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
            
        except Exception as e:
            print(f"❌ Fehler bei {steps} Steps: {e}")
            torch.cuda.empty_cache()
            gc.collect()
        
    print("\n🎉 QWEN Testbilder erfolgreich generiert! Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()