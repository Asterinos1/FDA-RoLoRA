"""
fix_table_float_layout.py
Fixes table float placement so Tables 5.1 and 5.2 never cut through text:
1. Tightens Section 5.1 items 4 and 5 so Section 5.1 finishes completely on Page 56.
2. Starts Section 5.2 with \clearpage so Section 5.2 heading, intro text, and Table 5.1 fit perfectly on Page 57.
3. Inserts \clearpage between Table 5.1 and Table 5.2 so Table 5.2 starts at the top of Page 58,
   followed immediately by the macroscopic inspection discussion on that same page.
"""

from pathlib import Path

eval_path = Path(r"C:\Users\GKaralis\Documents\GitHub\thesis-stuff\thesis\chapters\evaluation.tex")
text = eval_path.read_text(encoding="utf-8")

# 1. Format Item 4 and 5 inline to save vertical space on Page 56
old_items_4_5 = r"""    \item \textbf{GLUE Benchmark Corpus Standardization}\label{sec:eval_corpus}: 
    We evaluate downstream generalization across four GLUE benchmark tasks:
    \begin{itemize}
        \item \textbf{MNLI}: 3-class natural language inference ($392{,}702$ pairs).
        \item \textbf{QNLI}: Binary question-answering inference ($104{,}743$ pairs).
        \item \textbf{QQP}: Binary question paraphrase detection ($363{,}846$ pairs).
        \item \textbf{SST-2}: Binary sentiment classification ($67{,}349$ sentences).
    \end{itemize}
    Training sets are capped at $25{,}000$ samples to balance per-round compute, while validation sets are evaluated in full.

    \item \textbf{Statistical Data Partitioning Regimes}\label{sec:eval_regimes}: 
    Client data is partitioned via Dirichlet distribution sampling under two regimes:
    \begin{itemize}
        \item \textbf{Homogeneous (IID, $\alpha = 1000.0$)}: Clients hold uniform class distributions matching the global population.
        \item \textbf{Heterogeneous (Non-IID, $\alpha = 0.5$)}: Enforces severe label skew, where clients hold biased class subsets.
    \end{itemize}
    We evaluate cohorts of $C \in \{20, 50\}$ clients under full participation ($K = C$) per round."""

new_items_4_5 = r"""    \item \textbf{GLUE Benchmark Corpus Standardization}\label{sec:eval_corpus}: 
    We evaluate downstream generalization across four GLUE benchmarks: \textbf{MNLI} ($392{,}702$ sentence pairs), \textbf{QNLI} ($104{,}743$ QA pairs), \textbf{QQP} ($363{,}846$ paraphrase pairs), and \textbf{SST-2} ($67{,}349$ sentiment sentences). Training sets are capped at $25{,}000$ samples to balance per-round compute, while validation sets are evaluated in full.

    \item \textbf{Statistical Data Partitioning Regimes}\label{sec:eval_regimes}: 
    Client data is partitioned via Dirichlet distribution sampling across cohorts of $C \in \{20, 50\}$ under full participation ($K = C$). We evaluate two regimes: homogeneous consensus (IID, $\alpha = 1000.0$), where clients hold uniform class distributions, and heterogeneous label skew (Non-IID, $\alpha = 0.5$), where clients hold heavily biased class subsets."""

assert old_items_4_5 in text, "old_items_4_5 not found"
text = text.replace(old_items_4_5, new_items_4_5)

# 2. Add \clearpage before Section 5.2 and between Table 5.1 and Table 5.2
old_sec52_block = r"""% ==============================================================================
% SECTION 5.2: EXECUTIVE PERFORMANCE SCORECARD
% ==============================================================================
\section{Executive Performance Scorecard}
\label{sec:eval_scorecard}

To provide a clear, comprehensive synthesis across our entire benchmarking suite, this section presents executive performance scorecards directly contrasting FDA-RoLoRA configurations against the strongest fixed-schedule RoLoRA baseline across all 16 evaluated scenarios. To ensure readability and permit uncompressed, detailed presentation, the results are divided into two dedicated scorecards: Table~\ref{tab:scorecard_iid} evaluates the homogeneous consensus suite ($\alpha = 1000.0$), while Table~\ref{tab:scorecard_non_iid} evaluates the heterogeneous label skew suite ($\alpha = 0.5$).

Bold font denotes the overall superior peak accuracy for each scenario, while underlined font highlights the best-performing baseline. To capture statistical stability across random seeds, empirical sample standard deviations are reported in parentheses alongside mean peak accuracy values.

\input{tables/table_5_1_scorecard_iid.tex}

\input{tables/table_5_2_scorecard_noniid.tex}"""

new_sec52_block = r"""% ==============================================================================
% SECTION 5.2: EXECUTIVE PERFORMANCE SCORECARD
% ==============================================================================
\clearpage
\section{Executive Performance Scorecard}
\label{sec:eval_scorecard}

To provide a clear, comprehensive synthesis across our entire benchmarking suite, this section presents executive performance scorecards directly contrasting FDA-RoLoRA configurations against the strongest fixed-schedule RoLoRA baseline across all 16 evaluated scenarios. To ensure readability and permit uncompressed, detailed presentation, the results are divided into two dedicated scorecards: Table~\ref{tab:scorecard_iid} evaluates the homogeneous consensus suite ($\alpha = 1000.0$), while Table~\ref{tab:scorecard_non_iid} evaluates the heterogeneous label skew suite ($\alpha = 0.5$).

Bold font denotes the overall superior peak accuracy for each scenario, while underlined font highlights the best-performing baseline. To capture statistical stability across random seeds, empirical sample standard deviations are reported in parentheses alongside mean peak accuracy values.

\input{tables/table_5_1_scorecard_iid.tex}

\clearpage

\input{tables/table_5_2_scorecard_noniid.tex}"""

assert old_sec52_block in text, "old_sec52_block not found"
text = text.replace(old_sec52_block, new_sec52_block)

eval_path.write_text(text, encoding="utf-8")
print("Successfully updated evaluation.tex with clean layout.")
