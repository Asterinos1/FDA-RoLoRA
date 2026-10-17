import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

pos = text.find('Table 4:')
print("=== TEXT BEFORE TABLE 4 ===")
print(text[pos-1200:pos])
