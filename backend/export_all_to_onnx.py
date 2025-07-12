
import os
import sys
import torch
import subprocess
from pathlib import Path
import shutil

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

def check_mo_available():
    """Check if the Model Optimizer (mo) is available in PATH."""
    if shutil.which('mo') is None:
        print("❌ Model Optimizer (mo) not found in PATH. Please install OpenVINO and add 'mo' to your PATH.")
        print("See: https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html")
        return False
    return True

def export_vit():
    print("\n=== Exporting ViT to ONNX ===")
    try:
        from torchvision.models import vit_b_16
        weights_path = MODELS_DIR / 'vit' / 'vit_finetuned.pth'
        onnx_path = EXPORT_DIR / 'vit_b_16.onnx'
        expected_num_classes = 30  # <-- Set your dataset's class count here
        if not weights_path.exists():
            print(f"❌ ViT weights not found: {weights_path}\nPlease ensure the model checkpoint is present and named correctly.")
            return
        model = vit_b_16(weights=None)
        state_dict = torch.load(weights_path, map_location='cpu')
        # Automatically set the head size to match checkpoint
        if 'heads.head.weight' in state_dict:
            num_classes = state_dict['heads.head.weight'].shape[0]
            if num_classes != expected_num_classes:
                print(f"⚠️ ViT checkpoint has {num_classes} classes, but expected {expected_num_classes}. The model head will be reset and needs retraining.")
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, expected_num_classes)
            else:
                model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        else:
            print(f"⚠️ ViT checkpoint does not contain 'heads.head.weight'. Setting head to {expected_num_classes} classes.")
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, expected_num_classes)
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
        weights_path = MODELS_DIR / 'yolo' / 'yolo.pt'
        onnx_path = EXPORT_DIR / 'yolo.onnx'
        if not weights_path.exists():
            print(f"❌ YOLO weights not found: {weights_path}")
            return
        model = YOLO(str(weights_path))
        # Export ONNX to default location (next to weights)
        model.export(format='onnx', dynamic=True, imgsz=640, optimize=False, simplify=True, half=False)
        # Find the ONNX file (should be in weights_path.parent)
        default_onnx = weights_path.parent / 'yolo.onnx'
        if default_onnx.exists() and default_onnx != onnx_path:
            shutil.move(str(default_onnx), str(onnx_path))
            print(f"Moved YOLO ONNX to {onnx_path}")
        elif onnx_path.exists():
            print(f"YOLO ONNX already at {onnx_path}")
        else:
            print(f"❌ Could not find YOLO ONNX file after export.")
            return
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

def export_whisper_openvino():
    print("\n=== Exporting Whisper to OpenVINO IR using optimum-intel ===")
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        model_id = "openai/whisper-base"  # Change if you use a different Whisper model
        output_dir = EXPORT_DIR / 'openvino_whisper'
        print(f"Exporting {model_id} to {output_dir} ...")
        model = OVModelForSpeechSeq2Seq.from_pretrained(model_id, export=True)
        model.save_pretrained(str(output_dir))
        print(f"✅ Whisper exported to OpenVINO IR at {output_dir}")
        return output_dir
    except ImportError:
        print("❌ optimum-intel is not installed. Run: pip install optimum[openvino]")
    except Exception as e:
        print(f"❌ Whisper OpenVINO export failed: {e}")

def convert_to_openvino(onnx_path):
    print(f"\n=== Converting {onnx_path} to OpenVINO IR ===")
    xml_path = onnx_path.with_suffix('.xml')
    cmd = [
        'mo',
        '--input_model', str(onnx_path),
        '--output_dir', str(onnx_path.parent)
        # Removed '--data_type FP16' for compatibility
    ]
    try:
        run_cmd(cmd)
        print(f"✅ Converted to {xml_path}")
    except Exception as e:
        print(f"❌ OpenVINO conversion failed: {e}")

def main():
    print("\n🚀 Exporting all models to ONNX and OpenVINO IR (if possible)...")
    if not check_mo_available():
        print("Aborting OpenVINO IR conversion. ONNX export will still run.")
    onnx_paths = []
    for fn in [export_vit, export_yolo, export_blip, export_whisper, export_mistral]:
        onnx_path = fn()
        if onnx_path:
            onnx_paths.append(onnx_path)
    if check_mo_available():
        print("\n=== Converting all ONNX models to OpenVINO IR ===")
        for onnx_path in onnx_paths:
            convert_to_openvino(onnx_path)
        print("\n✅ All done!")
    else:
        print("\n⚠️  Skipped OpenVINO IR conversion due to missing Model Optimizer.")
    # Optionally, export Whisper to OpenVINO IR using optimum-intel
    export_whisper_openvino()

if __name__ == "__main__":
    main() 