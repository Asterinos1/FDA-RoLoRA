"""
apply_sec55_and_sec6_edits.py
1. Drops Section 5.5 from evaluation.tex completely.
2. Rewrites Section 6 in conclusion.tex according to exact user instructions.
"""

from pathlib import Path

# --- 1. Drop Section 5.5 from evaluation.tex ---
eval_path = Path(r"C:\Users\GKaralis\Documents\GitHub\thesis-stuff\thesis\chapters\evaluation.tex")
eval_text = eval_path.read_text(encoding="utf-8")

marker = "% ==============================================================================\n% SECTION 5.5: CROSS-REGIME SYNTHESIS AND GUIDELINES"
if marker not in eval_text:
    marker = r"\section{Cross-Regime Synthesis"

idx = eval_text.find(marker)
assert idx != -1, "Section 5.5 marker not found"

# Truncate evaluation.tex right before Section 5.5, stripping trailing whitespace
eval_clean = eval_text[:idx].rstrip() + "\n"
eval_path.write_text(eval_clean, encoding="utf-8")
print("Section 5.5 dropped completely from evaluation.tex.")

# --- 2. Rewrite conclusion.tex ---
concl_path = Path(r"C:\Users\GKaralis\Documents\GitHub\thesis-stuff\thesis\chapters\conclusion.tex")

concl_content = r"""\chapter{Conclusion and Future Work}
\label{chap:conclusion}

This thesis investigated the integration of Federated Dynamic Averaging (FDA) with Parameter-Efficient Low-Rank Adaptation (LoRA) in federated learning. By formulating and evaluating FDA-RoLoRA, this work addressed both spatial communication overhead and rigid temporal synchronization schedules under data heterogeneity.

\section{Synthesis of Results and Methodological Takeaways}
\label{sec:synthesis_results}

The formulation, implementation, and empirical evaluation of FDA-RoLoRA provide several key takeaways:

\begin{enumerate}
    \item \textbf{Preservation of Aggregation Linearity}: Dynamic averaging cannot be applied directly to standard federated LoRA. Updating both adapter matrices simultaneously creates a non-linear product aggregation error ($\frac{1}{N}\sum B_i A_i \neq (\frac{1}{N}\sum B_i)(\frac{1}{N}\sum A_i)$) that distorts client updates and misguides variance tracking. Alternating optimization updates only one factor ($A$ or $B$) per round, maintaining linear updates and allowing stable drift variance measurement.
    
    \item \textbf{Direct Subspace Drift Monitoring}: Tracking drift variance on low-rank coordinates avoids the heavy memory and network overhead of full-model gradient estimation. Using separate thresholds ($\theta_A, \theta_B$) for each factor allows the controller to adjust local epochs dynamically according to measured client divergence on the active matrix, preventing minority-class forgetting under skew.
    
    \item \textbf{Empirical Foundation Model Benchmarks}: Evaluations using \texttt{roberta-base} across GLUE tasks show that dynamic synchronization can achieve substantial communication round savings while preserving or improving validation accuracy under non-IID client skew.
\end{enumerate}

\section{Limitations}
\label{sec:limitations}

The findings of this work should be considered in light of several practical and experimental limitations:

\begin{enumerate}
    \item \textbf{Absence of Original RoLoRA Codebase}: The original implementation from the RoLoRA study \cite{chen2025robust} is not publicly available. Consequently, our work relied on an independent reimplementation from the paper's mathematical description, presenting significant engineering challenges in reconstructing and integrating its alternating dynamics into the FederatedScope-LLM framework.
    
    \item \textbf{Omission of Critical Hyperparameters}: The RoLoRA paper did not report key experimental hyperparameters, most notably the task-specific optimal client learning rates. Substantial exploratory tuning was therefore required to identify stable, competitive learning rate schedules for the baseline before comparative benchmarks could be conducted.
    
    \item \textbf{Hardware and Compute Constraints}: Limited local compute resources ($2\times$ NVIDIA A10 GPUs) restricted our experimental campaign to the 125M-parameter \texttt{roberta-base} model. While representative for masked language modeling and sequence classification, this prevented scaling to multi-billion parameter autoregressive Large Language Models (LLMs) and instruction-tuning workloads.
    
    \item \textbf{Simulated Federated Environment}: All benchmarks were executed within a multi-process simulated federated environment on a single server. Real-world network factors, including physical latency, fluctuating client bandwidth, and intermittent client dropouts, were not modeled.
\end{enumerate}

\section{Future Work}
\label{sec:future_work}

Building upon this work, several concrete directions for future research emerge:

\begin{enumerate}
    \item \textbf{Evaluation on Larger and More Diverse Models}: Expanding empirical evaluations beyond \texttt{roberta-base} to larger foundation models and modern autoregressive architectures (such as the LLaMA and Mistral families) to assess how dynamic coordinate variance scaling behaves on higher-capacity models and generative tasks.
    
    \item \textbf{Integration with Other Federated LoRA Architectures}: Exploring how the core mechanics of FDA can be adapted to alternative federated parameter-efficient methods, such as SLoRA \cite{babakniya2023slora} (sparse coordinate exchanges), FFA-LoRA \cite{sun2024improving} (frozen down-projections), and standard simultaneous FedLoRA to test whether coordinate monitoring can mitigate client drift in wider architectural settings.
    
    \item \textbf{Broader Experimental Scenarios for FDA-RoLoRA}: Conducting broader comparisons between FDA-RoLoRA and RoLoRA across varied federated conditions, including partial client participation per round, wider client cohort sizes ($C$), different client learning rates, and alternative matrix alternation orders (such as initializing with matrix $A$ first vs.\ matrix $B$ first).
    
    \item \textbf{Framework Integration into FederatedScope-LLM}: Implementing and contributing FDA natively into the widely used open-source FederatedScope-LLM framework \cite{kuang2023federatedscope}. This will make dynamic averaging readily accessible to the broader federated learning community and enable standardized benchmarking across emerging foundation model pipelines.
\end{enumerate}

In conclusion, this work demonstrates that combining Federated Dynamic Averaging with alternating low-rank adaptation effectively addresses both parameter transmission volume and rigid synchronization schedules. By establishing a mathematically consistent substrate for coordinate variance monitoring, FDA-RoLoRA offers a practical, communication-efficient approach to federated foundation model adaptation.
"""

concl_path.write_text(concl_content, encoding="utf-8")
print("Chapter 6 rewritten according to exact user instructions.")
