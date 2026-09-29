"""
Standalone script to fix a failed N-MNIST test-set download.

Usage:
    python scripts/fix_nmnist_testset.py

What it does:
  1. Checks whether the Test directory already exists and has the right number of files.
  2. If not, attempts to re-trigger Tonic's download (which skips Train/ if already present).
  3. If that fails, prints manual download instructions.

The training set (~965 MB) is unaffected — this script never touches Train/.
"""

import os
import sys

# Locate project root regardless of where this script is called from
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config import get_config

cfg = get_config()
data_root = cfg.data.data_root
test_dir  = os.path.join(data_root, "NMNIST", "Test")

EXPECTED_TEST_FILES = 10_000   # 1000 files per class × 10 classes

# -----------------------------------------------------------------------
# Step 1: check what's already on disk
# -----------------------------------------------------------------------
print("Checking test set on disk...")
existing = 0
if os.path.isdir(test_dir):
    for root, dirs, files in os.walk(test_dir):
        existing += len(files)

print(f"  Test directory : {test_dir}")
print(f"  Files found    : {existing:,}  (expected {EXPECTED_TEST_FILES:,})")

if existing == EXPECTED_TEST_FILES:
    print("\nTest set is already complete. No action needed.")
    sys.exit(0)

print("\nTest set incomplete — attempting download via Tonic...")

# -----------------------------------------------------------------------
# Step 2: retry Tonic's download
# -----------------------------------------------------------------------
try:
    import tonic
    # Instantiating with train=False triggers download for the test split only.
    # If Train/ already exists, Tonic will not re-download it.
    ds = tonic.datasets.NMNIST(save_to=data_root, train=False)
    print(f"Download succeeded. Dataset length: {len(ds):,}")
    sys.exit(0)

except RuntimeError as e:
    print(f"\nTonic download failed: {e}")
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
The Mendeley Data server is returning an error for the test-set URL.
Please download the test set manually:

  1. Open this URL in your browser:
     https://data.mendeley.com/datasets/468j46mzdv/2

  2. Download 'Test.zip' (the test-set archive, ~165 MB).

  3. Extract it so that your directory looks like:
       data/NMNIST/Test/
         0/   (1000 .bin files)
         1/   (1000 .bin files)
         ...
         9/   (1000 .bin files)

     The top-level folder inside the zip should be 'Test'.
     Place it directly at:  """ + test_dir + """

  4. Re-run this script to verify the file count.

Alternative mirrors (use only if Mendeley is still down):
  - Author's page: http://www.garrickorchard.com/datasets/n-mnist
    (direct link to N-MNIST zip containing both Train and Test)
    Warning: this is a large file; you only need to extract the Test/ folder.
""")
    sys.exit(1)
