import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Search for Figure 3 and Figure 8
for fig in ['Figure 3', 'Figure 8', 'Figure 2']:
    pos = text.find(fig)
    if pos != -1:
        print(f"=== {fig} ===")
        print(text[max(0, pos-200):pos+800])
