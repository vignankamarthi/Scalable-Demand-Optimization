"""
Diagnostic: inspect TimesFM model architecture.
Prints all attributes, children, and layer structure to stdout.
Output used to write the correct embedding extraction script.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.makedirs("results/timeseries", exist_ok=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import timesfm

# Load model
print("\nLoading TimesFM 2.5-200M...")
export_hf = os.environ.get("HF_HOME", "")
print(f"HF_HOME: {export_hf}")

tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
print("Model loaded.")

# Inspect top-level object
print("\n=== TOP-LEVEL (tfm) ===")
print(f"Type: {type(tfm)}")
print(f"Public attrs: {[a for a in dir(tfm) if not a.startswith('_')]}")

# Inspect .model if it exists
if hasattr(tfm, "model"):
    model = tfm.model
    print(f"\n=== tfm.model ===")
    print(f"Type: {type(model)}")
    print(f"Public attrs: {[a for a in dir(model) if not a.startswith('_')]}")

    print(f"\n=== tfm.model named_children ===")
    for name, module in model.named_children():
        print(f"  {name}: {type(module).__name__}")
        # One level deeper
        for sub_name, sub_module in module.named_children():
            print(f"    {name}.{sub_name}: {type(sub_module).__name__}")

    print(f"\n=== tfm.model named_modules (first 30) ===")
    for i, (name, module) in enumerate(model.named_modules()):
        if i >= 30:
            print(f"  ... ({i} modules shown, more exist)")
            break
        print(f"  {name}: {type(module).__name__}")

# Check for forward method signature
print(f"\n=== Forward method ===")
import inspect
if hasattr(tfm, "model") and hasattr(tfm.model, "forward"):
    sig = inspect.signature(tfm.model.forward)
    print(f"tfm.model.forward{sig}")

# Try a dummy forward pass to see output structure
print(f"\n=== Dummy forecast ===")
try:
    config = timesfm.ForecastConfig(
        max_context=64,
        max_horizon=1,
        per_core_batch_size=1,
    )
    tfm.compile(config)
    import numpy as np
    dummy = [np.random.randn(64).astype(np.float32)]
    result = tfm.forecast(horizon=1, inputs=dummy)
    print(f"forecast() return type: {type(result)}")
    if isinstance(result, tuple):
        for i, r in enumerate(result):
            if hasattr(r, "shape"):
                print(f"  result[{i}]: shape={r.shape}, dtype={r.dtype}")
            else:
                print(f"  result[{i}]: type={type(r)}")
except Exception as e:
    print(f"Dummy forecast failed: {e}")

# Save structured output for local analysis
output = {
    "tfm_type": str(type(tfm)),
    "tfm_attrs": [a for a in dir(tfm) if not a.startswith("_")],
}
if hasattr(tfm, "model"):
    output["model_type"] = str(type(tfm.model))
    output["model_attrs"] = [a for a in dir(tfm.model) if not a.startswith("_")]
    output["model_children"] = {
        name: str(type(module).__name__)
        for name, module in tfm.model.named_children()
    }
    output["model_modules_first30"] = {
        name: str(type(module).__name__)
        for i, (name, module) in enumerate(tfm.model.named_modules())
        if i < 30
    }

probe_path = "results/timeseries/timesfm_probe.json"
with open(probe_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nProbe saved to {probe_path}")
