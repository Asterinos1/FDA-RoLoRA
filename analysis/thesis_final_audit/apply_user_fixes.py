"""
apply_user_fixes.py
Applies the three requested surgical edits to thesis/chapters/evaluation.tex:
1. Remove parentheses and contents at the end of Section 5 introduction.
2. Fix roberta-base line overflow in Section 5.1 item 2.
3. Replace item 4 training ceiling sentence with the exact requested phrase.
"""

from pathlib import Path

eval_path = Path(r"C:\Users\GKaralis\Documents\GitHub\thesis-stuff\thesis\chapters\evaluation.tex")
text = eval_path.read_text(encoding="utf-8")

# 1. Section 5 intro parentheses removal
old_intro_suffix = r"established in Chapter~\ref{chap:methodology} (operating with decoupled factor thresholds, variance-proportional scaling, and no heuristic moving-average lags)."
new_intro_suffix = r"established in Chapter~\ref{chap:methodology}."
assert old_intro_suffix in text, "old_intro_suffix not found"
text = text.replace(old_intro_suffix, new_intro_suffix)

# 2. Section 5.1 item 2 roberta-base overflow fix
old_item2_start = "    \\item \\textbf{Foundation Model \\& Low-Rank Parameterization}\\label{sec:eval_model_param}: \n    We evaluate the \\texttt{roberta-base} model"
new_item2_start = "    \\item \\textbf{Foundation Model \\& Low-Rank Parameterization}\\label{sec:eval_model_param}: \n    Our evaluations employ the \\texttt{roberta\\allowbreak-base} model"

assert old_item2_start in text, "old_item2_start not found"
text = text.replace(old_item2_start, new_item2_start)

# 3. Section 5.1 item 4 training ceiling sentence simplification
old_item4_sentence = r"Training sets exceeding $25{,}000$ examples are capped at $25{,}000$ samples ($\text{MAX\_SAMPLES} = 25{,}000$) to balance per-round compute, while validation sets are evaluated in full ($100\%$)."
new_item4_sentence = r"Training sets are capped at $25{,}000$ samples to balance per-round compute, while validation sets are evaluated in full."

assert old_item4_sentence in text, "old_item4_sentence not found"
text = text.replace(old_item4_sentence, new_item4_sentence)

eval_path.write_text(text, encoding="utf-8")
print("Successfully applied user fixes to evaluation.tex")
