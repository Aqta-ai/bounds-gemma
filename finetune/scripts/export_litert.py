"""
Export the merged Gemma 4 E2B PHI fine-tune to LiteRT-LM .task format
for WebLLM / MediaPipe LLM Inference in the browser.

Inputs:
  - bounds-gemma-e2b-phi-ft/merged/  (HuggingFace transformers model dir)

Outputs:
  - bounds-gemma-e2b-phi-ft.task     (LiteRT-LM bundle, ~1.5-2 GB)
  - bounds-gemma-e2b-phi-ft-web.task (Web-optimised variant, same size)

Usage on Colab T4:
  !pip install ai-edge-torch ai-edge-quantizer mediapipe-model-maker
  !python finetune/scripts/export_litert.py

LiteRT-LM is the runtime format MediaPipe LLM Inference and WebLLM
consume. Conversion takes ~10-15 minutes on T4. The output .task is
self-contained: tokenizer, weights, and metadata in one bundle, ready
to drop into a Hugging Face repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MERGED_DIR = Path(os.environ.get("MERGED_DIR", "bounds-gemma-e2b-phi-ft/merged"))
OUTPUT_TASK = Path(os.environ.get("OUTPUT_TASK", "bounds-gemma-e2b-phi-ft.task"))
WEB_OUTPUT_TASK = Path(os.environ.get("WEB_OUTPUT_TASK", "bounds-gemma-e2b-phi-ft-web.task"))


def export_litert():
    """Run the AI Edge Torch export pipeline."""
    if not MERGED_DIR.exists():
        raise FileNotFoundError(
            f"Merged model missing: {MERGED_DIR}. Run train.py first."
        )

    print(f"Merged model:     {MERGED_DIR}")
    print(f"LiteRT .task:     {OUTPUT_TASK}")
    print(f"Web-opt .task:    {WEB_OUTPUT_TASK}")
    print()

    # Lazy imports
    try:
        import ai_edge_torch
        from ai_edge_torch.generative.examples.gemma3 import gemma3
        from ai_edge_torch.generative.utilities import converter
    except ImportError as err:
        print("Missing dependency. Install with:")
        print("  pip install ai-edge-torch ai-edge-quantizer")
        sys.exit(1)

    # Build the Gemma-4 architecture skeleton and load weights from the
    # merged HF model. The converter then traces the model with example
    # inputs and emits an int4-quantised .task file.
    print("[1/3] Loading merged weights into Gemma 4 architecture…")
    pytorch_model = gemma3.build_model(str(MERGED_DIR), kv_cache_max_len=2048)

    print("[2/3] Converting to LiteRT-LM (int4 quantisation)…")
    converter.convert_to_tflite(
        pytorch_model,
        tflite_path=str(OUTPUT_TASK),
        prefill_seq_lens=[128, 512, 1024],
        quantize="dynamic_int4",
        export_config=None,
    )
    print(f"  Wrote {OUTPUT_TASK} ({OUTPUT_TASK.stat().st_size / (1024**3):.2f} GB)")

    # The "web" variant is the same .task with a smaller kv-cache for
    # the browser-WebGPU path. MediaPipe LLM Inference accepts either,
    # but the web one is gentler on browser memory.
    print("[3/3] Converting web-optimised variant…")
    pytorch_model_web = gemma3.build_model(str(MERGED_DIR), kv_cache_max_len=1024)
    converter.convert_to_tflite(
        pytorch_model_web,
        tflite_path=str(WEB_OUTPUT_TASK),
        prefill_seq_lens=[128, 512],
        quantize="dynamic_int4",
        export_config=None,
    )
    print(f"  Wrote {WEB_OUTPUT_TASK} ({WEB_OUTPUT_TASK.stat().st_size / (1024**3):.2f} GB)")

    print()
    print("Done. Next steps:")
    print(f"  1. Verify file size is ~1.5 GB (web) or ~2 GB (full). If much")
    print(f"     larger, the int4 quantisation probably didn't apply.")
    print(f"  2. Push to HuggingFace as a public model repo:")
    print(f"       huggingface-cli upload Aqta-ai/bounds-gemma-e2b-phi-ft {OUTPUT_TASK}")
    print(f"       huggingface-cli upload Aqta-ai/bounds-gemma-e2b-phi-ft {WEB_OUTPUT_TASK}")
    print(f"  3. Update bounds-gemma to point at the new URL:")
    print(f"       NEXT_PUBLIC_GEMMA_E2B_URL=https://huggingface.co/Aqta-ai/bounds-gemma-e2b-phi-ft/resolve/main/bounds-gemma-e2b-phi-ft-web.task")


if __name__ == "__main__":
    export_litert()
