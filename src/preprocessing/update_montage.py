import sys
from pathlib import Path

montage_file = Path(__file__).resolve().parents[2] / 'src/preprocessing/montage.py'
content = montage_file.read_text()

unipolar_code = 
