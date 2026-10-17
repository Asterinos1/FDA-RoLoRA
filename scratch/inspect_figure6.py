import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

pos = text.find('Figure 6:')
if pos != -1:
    print("=== FIGURE 6 TEXT ===")
    print(text[max(0, pos-200):pos+800])
