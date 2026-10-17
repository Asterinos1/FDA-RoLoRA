import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

pos = text.find('Table 4:')
if pos != -1:
    print("=== TABLE 4 TEXT ===")
    print(text[max(0, pos-400):pos+1200])
