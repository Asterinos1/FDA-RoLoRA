import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Search for partitioning / data distribution description in Section 5 (Experiments)
pos = text.find('5 EXPERIMENTS')
if pos == -1:
    pos = text.find('EXPERIMENTS')
print("=== EXPERIMENTAL SETUP ===")
print(text[pos:pos+3500])
