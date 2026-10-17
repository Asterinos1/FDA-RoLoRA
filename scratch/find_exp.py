import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

for m in re.finditer(r'([0-9]\s+Experiments?|Section\s+[0-9].*Experiment)', text, re.IGNORECASE):
    pos = m.start()
    print("Match:", text[pos:pos+1500])
    print("="*60)
