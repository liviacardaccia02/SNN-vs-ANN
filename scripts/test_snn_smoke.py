"""Quick smoke test to diagnose SNN construction and forward pass outside pytest."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from spikingjelly.activation_based import functional

from src.config import get_config
from src.models.snn import SNN

cfg = get_config()

print("--- 1. Construct SNN ---")
snn = SNN(tau=cfg.snn.tau, v_threshold=cfg.snn.v_threshold)
print("OK")

print("\n--- 2. Call functional.reset_net(snn) ---")
try:
    functional.reset_net(snn)
    print("OK (no recursion)")
except RecursionError as e:
    print(f"RECURSION ERROR from functional.reset_net: {e}")

print("\n--- 3. Call snn.reset() ---")
try:
    snn.reset()
    print("OK (no recursion)")
except RecursionError as e:
    print(f"RECURSION ERROR from snn.reset(): {e}")

print("\n--- 4. Forward pass T=4, B=2 ---")
try:
    functional.reset_net(snn)
    x = torch.zeros(4, 2, 2, 34, 34)   # [T, B, C, H, W]
    with torch.no_grad():
        out = snn(x)
    print(f"OK — output shape: {tuple(out.shape)}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n--- 5. Forward pass T=16, B=4 ---")
try:
    functional.reset_net(snn)
    x = torch.rand(16, 4, 2, 34, 34)
    with torch.no_grad():
        out = snn(x)
    print(f"OK — output shape: {tuple(out.shape)}, finite: {torch.isfinite(out).all().item()}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n--- 6. Thresholds ---")
print(snn.get_thresholds())
