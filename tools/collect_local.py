#!/usr/bin/env python3
"""Export source notebooks/code and optional HTML, excluding data/models/caches."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assignment', type=int, choices=[1, 2, 3], required=True)
parser.add_argument('--html', action='store_true')
args = parser.parse_args()
folder = ROOT / f'assignments/assignment{args.assignment}/assignment{args.assignment}'
output = folder / 'results/submission'
output.mkdir(parents=True, exist_ok=True)
notebooks = sorted(p for p in folder.glob('*.ipynb') if p.name != 'collect_submission.ipynb')
code = [p for p in (folder / 'cs231n').rglob('*') if p.suffix in {'.py', '.pyx'}
        and not any(x in p.relative_to(folder).parts for x in ['datasets', '__pycache__', 'build', 'exp'])]
with zipfile.ZipFile(output / f'a{args.assignment}_code.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in notebooks + code:
        z.write(p, p.relative_to(folder))
if args.html:
    for p in notebooks:
        subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'html',
                        '--output-dir', str(output), str(p)], check=True)
print('导出到：', output, '（无数据、模型和 TeX 安装；HTML 可用浏览器另存 PDF）')
