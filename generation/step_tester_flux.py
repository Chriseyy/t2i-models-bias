"""
step_tester_flux_dev.py
=======================
Ablation Study für FLUX.2-dev (4-Bit).
Zeigt, ab welchem Denoising-Schritt (Point of No Return) 
sich die Semantik (Bias/Stereotyp) bei FLUX nicht mehr ändert.
"""

import os
import torch
import time
from pathlib import Path
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from dotenv import load_dotenv

# 1. Setup & Pfade
load_dotenv()
OUTPUT_DIR = Path("outputs/step_test/flux_dev")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Test-Parameter
PROMPT = "a photo of a beautiful person"
PROMPT = "a cat sitting on a windowsill"   # https://huggingface.co/spaces/multimodalart/flux2-quantization   https://huggingface.co/blog/flux-2#lora-fine-tuning
SEED = 103
SEED = 42 
# Da FLUX.2-dev 28 Steps empfiehlt, testen wir diese Spanne:
STEPS_TO_TEST = [1, 2, 5, 10, 20, 25, 28, 30, 40, 50]

def main():
    print("=" * 60)
    print("🔬 KI-ZEITLUPE STARTEN: FLUX.2-dev (4-Bit Ablation Study)")
    print("=" * 60)
    
    model_id = "diffusers/FLUX.2-dev-bnb-4bit"
    dtype = torch.bfloat16

    print(f"Lade lokales 4-Bit Modell: {model_id}")
    
    # 3. Text-Encoder & Transformer sicher laden (wie in deinem run_flux.py)
    print("   -> Lade 4-Bit Text-Encoder...")
    text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype, device_map="cpu"
    )
    
    print("   -> Lade 4-Bit Transformer...")
    dit = AutoModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=dtype, device_map="cpu"
    )
    
    print("   -> Baue Pipeline zusammen...")
    pipe = Flux2Pipeline.from_pretrained(
        model_id, text_encoder=text_encoder, transformer=dit, torch_dtype=dtype
    )
    
    # CPU Offload belassen wir hier als Sicherheitsnetz für das riesige Dev-Modell
    pipe.enable_model_cpu_offload()
    
    # === ANTI-FREEZE SCHUTZ FÜR DEINEN PC ===
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print("\n🚀 Starte Generierungen...")
    
    # 4. Loop durch die Steps
    for steps in STEPS_TO_TEST:
        # Extrem wichtig: Generator bei JEDEM Step neu mit Seed 101 starten!
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        start_time = time.time()
        
        image = pipe(
            prompt=PROMPT,
            num_inference_steps=steps,
            # guidance_scale=4.0,
            guidance_scale=2.5,
            generator=generator,
            height=1024,    
            width=1024
        ).images[0]
        
        gen_time = time.time() - start_time
        
        # 5. Speichern
        filename = OUTPUT_DIR / f"doctor_seed{SEED}_{steps:02d}steps.png"
        image.save(filename)
        
        print(f"✅ Bild mit {steps:02d} Steps gespeichert! (Dauer: {gen_time:.1f}s) -> {filename.name}")
        
    print("\n🎉 FLUX.2-dev Testbilder generiert! Schau in den Ordner:", OUTPUT_DIR)

if __name__ == "__main__":
    main()