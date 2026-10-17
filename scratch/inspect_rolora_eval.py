import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

with open('sources/rolora_verr4.txt', 'r', encoding='utf-8', errors='replace') as f:
    text = f.read()

# Let's search for "evaluate" or "evaluation" or "test set" or "validation"
for m in re.finditer(r'(evaluate|evaluation|test set|validation)', text, re.IGNORECASE):
    pos = m.start()
    snippet = text[max(0, pos-100):pos+300].replace('\n', ' ')
    print(f"- {snippet}")
