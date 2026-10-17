import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

pos = text.find('Table 1:')
print("=== TABLE 1 TEXT ===")
print(text[max(0, pos-400):pos+1200])

pos_app = text.find('Table 7:')
if pos_app != -1:
    print("=== TABLE 7 TEXT ===")
    print(text[max(0, pos_app-400):pos_app+1200])
