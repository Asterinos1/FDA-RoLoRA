"""
shorten_sec51.py
Simplifies Section 5.1 introduction to one simple line, and shortens each item slightly.
"""

from pathlib import Path

eval_path = Path(r"C:\Users\GKaralis\Documents\GitHub\thesis-stuff\thesis\chapters\evaluation.tex")
text = eval_path.read_text(encoding="utf-8")

# Define target to replace
start_marker = "% ==============================================================================\n% SECTION 5.1: EXPERIMENTAL SETUP AND BENCHMARK PROTOCOL"
end_marker = "% ==============================================================================\n% SECTION 5.2: EXECUTIVE PERFORMANCE SCORECARD"

idx_start = text.find(start_marker)
idx_end = text.find(end_marker)

assert idx_start != -1, "start_marker not found"
assert idx_end != -1, "end_marker not found"

replacement = r"""% ==============================================================================
% SECTION 5.1: EXPERIMENTAL SETUP AND BENCHMARK PROTOCOL
% ==============================================================================
\section{Experimental Setup and Benchmark Protocol}
\label{sec:eval_setup}

This section describes our experimental setup and evaluation protocol across seven key areas:

\begin{enumerate}
    \item \textbf{Deployment Infrastructure \& Hardware Testbed}\label{sec:eval_infrastructure}: 
    All experiments ran on the dedicated Polytechnix GPU server at the SoftNet Lab of the Technical University of Crete (Section~\ref{sec:impl_deployment}). Using this uniform hardware platform ensured fair comparisons across all runs without queue delays or shared-resource interference.

    \item \textbf{Foundation Model \& Low-Rank Parameterization}\label{sec:eval_model_param}: 
    We evaluate the \texttt{roberta-base} model ($125\text{M}$ parameters, 12 layers, hidden size 768) with frozen backbone weights $W_0$. Following normalized depth mapping from RoLoRA's $\mathcal{P}_2$ budget \cite{chen2025robust}, LoRA adapters ($r = 4, \alpha = 8$) are added to the query ($W_q$) and value ($W_v$) projections of the top 5 blocks $\{7, \dots, 11\}$ for MNLI, QNLI, and QQP (10 active modules), and the top 3 blocks $\{9, 10, 11\}$ for SST-2 (6 active modules). In alternating optimization, each round updates and communicates strictly one factor matrix ($A$ or $B$). Under 10 active modules, bi-directional Float32 communication per client per round is:
    \begin{equation}
        \text{Payload}_{\text{client}} = 2 \times 10 \times (768 \times 4) \times 4\,\text{bytes} = 245{,}760\,\text{bytes} = 240\,\text{KB} \; (0.234375\,\text{MB}) \label{eq:payload_formulation}
    \end{equation}
    consisting of $120\,\text{KB}$ downlink broadcast and $120\,\text{KB}$ uplink upload (totaling $4.69\,\text{MB}$ per round for $C = 20$, and $11.72\,\text{MB}$ for $C = 50$).

    \item \textbf{Capacity Context Relative to Published RoLoRA Benchmarks}\label{sec:eval_capacity}: 
    While Chen et al. \cite{chen2025robust} evaluated the 355M-parameter \texttt{roberta-large} model, our work uses the 125M-parameter \texttt{roberta-base} to support extensive multi-seed sweeps within our institutional compute budget ($2 \times$ NVIDIA A10 GPUs). Absolute validation scores are therefore lower than those for \texttt{roberta-large}. All comparisons are strictly head-to-head on the identical \texttt{roberta-base} backbone, isolating the algorithmic impact of dynamic variance tracking.

    \item \textbf{GLUE Benchmark Corpus Standardization}\label{sec:eval_corpus}: 
    We evaluate downstream generalization across four GLUE benchmark tasks:
    \begin{itemize}
        \item \textbf{MNLI}: 3-class natural language inference ($392{,}702$ pairs).
        \item \textbf{QNLI}: Binary question-answering inference ($104{,}743$ pairs).
        \item \textbf{QQP}: Binary question paraphrase detection ($363{,}846$ pairs).
        \item \textbf{SST-2}: Binary sentiment classification ($67{,}349$ sentences).
    \end{itemize}
    Training sets exceeding $25{,}000$ examples are capped at $25{,}000$ samples ($\text{MAX\_SAMPLES} = 25{,}000$) to balance per-round compute, while validation sets are evaluated in full ($100\%$).

    \item \textbf{Statistical Data Partitioning Regimes}\label{sec:eval_regimes}: 
    Client data is partitioned via Dirichlet distribution sampling under two regimes:
    \begin{itemize}
        \item \textbf{Homogeneous (IID, $\alpha = 1000.0$)}: Clients hold uniform class distributions matching the global population.
        \item \textbf{Heterogeneous (Non-IID, $\alpha = 0.5$)}: Enforces severe label skew, where clients hold biased class subsets.
    \end{itemize}
    We evaluate cohorts of $C \in \{20, 50\}$ clients under full participation ($K = C$) per round.

    \item \textbf{Hyperparameter Matrix \& Controller Horizons}\label{sec:eval_hyperparameters}: 
    Local training uses SGD with client learning rate $\eta_c = 0.1$ and batch size $B = 32$ ($B = 64$ for SST-2, matching \cite{chen2025robust}), while the server applies SGD with learning rate $\eta_s = 1.0$. Static RoLoRA baselines are tested across fixed epochs $E \in \{5, 10, 15, 20\}$. FDA-RoLoRA adapts local epochs within $[e_{\min}, e_{\max}] = [1, 20]$ using scaling parameter $k \in \{0.5, 1.0\}$. An accuracy-gated early stopping protocol \label{sec:eval_early_stopping} terminates runs when validation accuracy plateaus ($W=3, P=3$ in IID; $W=5, P=5$ in non-IID).

    \item \textbf{Evaluation Metrics \& Cost Formulation}\label{sec:eval_metrics}: 
    Performance is evaluated across six complementary metrics:
    \begin{itemize}
        \item \textbf{Peak Validation Accuracy ($\text{Acc}_{\text{peak}}$)}: Maximum global validation accuracy attained across all rounds.
        \item \textbf{Final Validation Accuracy ($\text{Acc}_{\text{final}}$)}: Validation accuracy recorded at training termination.
        \item \textbf{Communication Rounds ($R$)}: Total rounds executed before early stopping or reaching $R_{\max}$.
        \item \textbf{Cumulative Network Communication (Total MB)}: Total Float32 adapter payload transferred (Equation~\ref{eq:payload_formulation}).
        \item \textbf{Realized Mean Epochs ($\bar{e}$)}: Average local epochs executed per round, decomposed into $\bar{e}_A$ and $\bar{e}_B$.
        \item \textbf{Subspace Drift Variance ($\bar{\nu}$)}: Mean coordinate variance $\sigma_t^2$ measuring inter-client drift prior to aggregation.
    \end{itemize}
\end{enumerate}

"""

new_text = text[:idx_start] + replacement + text[idx_end:]
eval_path.write_text(new_text, encoding="utf-8")
print("Section 5.1 successfully updated.")
