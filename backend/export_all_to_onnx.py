
import os
import sys
import torch
import subprocess
from pathlib import Path

# ========== CONFIG ==========
MODELS_DIR = Path(__file__).parent / 'models'
EXPORT_DIR = Path(__file__).parent / 'openvino_models'
EXPORT_DIR.mkdir(exist_ok=True)

# ========== UTILS ==========
def run_cmd(cmd):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")

def export_vit():
    print("\n=== Exporting ViT to ONNX ===")
    try:
        from torchvision.models import vit_b_16
        weights_path = MODELS_DIR / 'vit' / 'vit_finetuned.pth'
        onnx_path = EXPORT_DIR / 'vit_b_16.onnx'
        if not weights_path.exists():
            print(f"❌ ViT weights not found: {weights_path}")
            return
        model = vit_b_16(weights=None)
        state_dict = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        dummy = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy, onnx_path, input_names=['input'], output_names=['output'],
                          opset_version=14, dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
        print(f"✅ ViT exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ ViT export failed: {e}")

def export_yolo():
    print("\n=== Exporting YOLO to ONNX ===")
    try:
        from ultralytics import YOLO
        weights_path = MODELS_DIR / 'yolo' / 'best.pt'
        onnx_path = EXPORT_DIR / 'yolo.onnx'
        if not weights_path.exists():
            print(f"❌ YOLO weights not found: {weights_path}")
            return
        model = YOLO(str(weights_path))
        model.export(format='onnx', dynamic=True, imgsz=640, optimize=False, simplify=True, half=False, device='cpu',
                     output=str(onnx_path))
        print(f"✅ YOLO exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ YOLO export failed: {e}")

def export_blip():
    print("\n=== Exporting BLIP to ONNX ===")
    try:
        from transformers import BlipForConditionalGeneration
        weights_path = MODELS_DIR / 'blip-image-captioning-base'
        onnx_path = EXPORT_DIR / 'blip.onnx'
        if not weights_path.exists():
            print(f"❌ BLIP weights not found: {weights_path}")
            return
        model = BlipForConditionalGeneration.from_pretrained(str(weights_path))
        model.eval()
        dummy = {
            'pixel_values': torch.randn(1, 3, 224, 224),
            'input_ids': torch.ones(1, 10, dtype=torch.long)
        }
        torch.onnx.export(model, (dummy['pixel_values'], dummy['input_ids']), onnx_path,
                          input_names=['pixel_values', 'input_ids'], output_names=['output'],
                          opset_version=14, dynamic_axes={'pixel_values': {0: 'batch'}, 'output': {0: 'batch'}})
        print(f"✅ BLIP exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ BLIP export failed: {e}")

def export_whisper():
    print("\n=== Exporting Whisper to ONNX ===")
    try:
        from transformers import WhisperForConditionalGeneration
        weights_path = MODELS_DIR / 'whisper'
        onnx_path = EXPORT_DIR / 'whisper.onnx'
        if not weights_path.exists():
            print(f"❌ Whisper weights not found: {weights_path}")
            return
        model = WhisperForConditionalGeneration.from_pretrained(str(weights_path))
        model.eval()
        dummy = {
            'input_features': torch.randn(1, 80, 3000),
            'decoder_input_ids': torch.ones(1, 10, dtype=torch.long)
        }
        torch.onnx.export(model, (dummy['input_features'], dummy['decoder_input_ids']), onnx_path,
                          input_names=['input_features', 'decoder_input_ids'], output_names=['output'],
                          opset_version=14, dynamic_axes={'input_features': {0: 'batch'}, 'output': {0: 'batch'}})
        print(f"✅ Whisper exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ Whisper export failed: {e}")

def export_mistral():
    print("\n=== Exporting Mistral to ONNX ===")
    try:
        from transformers import AutoModelForCausalLM
        weights_path = MODELS_DIR / 'mistral'
        onnx_path = EXPORT_DIR / 'mistral.onnx'
        if not weights_path.exists():
            print(f"❌ Mistral weights not found: {weights_path}")
            return
        model = AutoModelForCausalLM.from_pretrained(str(weights_path))
        model.eval()
        dummy = torch.ones(1, 32, dtype=torch.long)
        torch.onnx.export(model, dummy, onnx_path, input_names=['input_ids'], output_names=['output'],
                          opset_version=14, dynamic_axes={'input_ids': {0: 'batch'}, 'output': {0: 'batch'}})
        print(f"✅ Mistral exported to {onnx_path}")
        return onnx_path
    except Exception as e:
        print(f"❌ Mistral export failed: {e}")

def convert_to_openvino(onnx_path):
    print(f"\n=== Converting {onnx_path} to OpenVINO IR ===")
    xml_path = onnx_path.with_suffix('.xml')
    cmd = [
        'mo',
        '--input_model', str(onnx_path),
        '--output_dir', str(onnx_path.parent),
        '--data_type', 'FP16'
    ]
    try:
        run_cmd(cmd)
        print(f"✅ Converted to {xml_path}")
    except Exception as e:
        print(f"❌ OpenVINO conversion failed: {e}")

def main():
    print("\n🚀 Exporting all models to ONNX and OpenVINO IR (if possible)...")
    onnx_paths = []
    for fn in [export_vit, export_yolo, export_blip, export_whisper, export_mistral]:
        onnx_path = fn()
        if onnx_path:
            onnx_paths.append(onnx_path)
    print("\n=== Converting all ONNX models to OpenVINO IR ===")
    for onnx_path in onnx_paths:
        convert_to_openvino(onnx_path)
    print("\n✅ All done!")

if __name__ == "__main__":
    main() 