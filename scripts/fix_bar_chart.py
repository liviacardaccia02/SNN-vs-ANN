"""Fix the bar-chart line in the Stage 1 notebook."""
import pathlib

p = pathlib.Path("notebooks/01_env_and_dataset.py")
txt = p.read_text(encoding="utf-8")
lines = txt.splitlines()

fixed = 0
for i, line in enumerate(lines):
    if "bar =" in line and "int(cnt" in line:
        lines[i] = '        bar = "#" * int(cnt / counts.max() * 30)'
        print(f"Fixed line {i+1}: {lines[i].strip()}")
        fixed += 1

if fixed == 0:
    print("WARNING: bar= line not found — printing lines containing 'bar':")
    for i, line in enumerate(lines):
        if "bar" in line and "int" in line:
            print(f"  {i+1}: {repr(line)}")
else:
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {p} with {fixed} fix(es).")
