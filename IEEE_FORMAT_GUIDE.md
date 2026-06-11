# How to Write Your Report in IEEE Format — A Practical Guide

This guide teaches you the IEEE conference-paper format and shows exactly how to
turn [report.md](report.md) into a submission that satisfies the course rule
*"報告書電子檔(IEEE格式 7–8 頁)"*. It is written for someone who has never used
the template before.

---

## 1. What "IEEE 格式" actually means

"IEEE format" = the **IEEE conference proceedings template**, class file
`IEEEtran`. Its visual signature:

- **Two columns**, justified text, single line spacing.
- **Times New Roman / Times**, body text **10 pt**.
- US-Letter paper, narrow margins (~0.625 in / 1.59 cm).
- A specific **title block** (title, authors, affiliations), then **Abstract**
  (bold run-in, italic-ish), then **Index Terms**, then the body.
- **Roman-numeral section headings** (I, II, III…), small-caps style.
- **Figures/tables** float to the top/bottom of a column or span both columns;
  captions are **"Fig. 1."** (below figures) and **"TABLE I"** (above tables).
- **Numbered citations** in square brackets `[1]`, listed in order of first
  appearance.

You do **not** have to use LaTeX — IEEE also ships a Word template — but LaTeX
(via **Overleaf**, free in the browser) is the path of least pain because it
handles two-column floating and references automatically.

---

## 2. Fastest path: Overleaf (recommended)

1. Create a free account at overleaf.com.
2. **New Project → Templates → search "IEEE Conference"** (the official
   *IEEEtran* template appears). Or **New Project → Upload Project** and drop in
   `IEEEtran.cls` from the IEEE Author Center.
3. You get a `main.tex` that starts like this:

```latex
\documentclass[conference]{IEEEtran}
\usepackage{cite}          % nice [1],[2] handling
\usepackage{amsmath,amssymb}
\usepackage{graphicx}      % figures
\usepackage{booktabs}      % good tables
\usepackage{url}
\begin{document}
\title{Reproduction and Architectural Ablation of PPO and DQN
       for Discrete-Action Autonomous Highway Driving}
\author{\IEEEauthorblockN{Your Name}
        \IEEEauthorblockA{Department, University\\ email}}
\maketitle
\begin{abstract}
... your abstract ...
\end{abstract}
\begin{IEEEkeywords}
Reinforcement learning, PPO, DQN, self-attention, autonomous driving.
\end{IEEEkeywords}
\section{Introduction}
...
\end{document}
```

4. Paste each section of `report.md` into the matching `\section{}`. Convert
   Markdown to LaTeX as you go (see §5 cheat-sheet).
5. Click **Recompile** → PDF. Check it is **7–8 pages**.

---

## 3. The required section structure (and what goes in each)

IEEE papers follow a near-universal skeleton. `report.md` is already organised
this way, so you can map 1-to-1:

| # | IEEE Section | What it must contain | In `report.md` |
|---|---|---|---|
| — | **Title + Authors** | Concise, specific title; your name/affiliation | top |
| — | **Abstract** (≤ ~250 words) | Problem, method, key result, 1 number | "Abstract" |
| — | **Index Terms** | 4–6 keywords | "Index Terms" |
| I | **Introduction** | Motivation, the gap, your contributions (bullet list) | §I |
| II | **Background / Related Work** | Prior methods you build on, with `[refs]` | §II |
| III | **Problem Formulation** | MDP, state/action/reward, formal setup | §III |
| IV | **Methods** | Your algorithms + architectures, equations, tables | §IV |
| V | **Experimental Setup** | Hyperparameters, seeds, protocol, hardware | §V |
| VI | **Results** | Figures + tables + objective description | §VI |
| VII | **Discussion** | What the numbers *mean*, honestly | §VII |
| VIII | **Limitations** | Threats to validity (shows maturity) | §VIII |
| IX | **Conclusion** | 1 paragraph recap + future work | §IX |
| — | **References** | IEEE numbered style | "References" |

**Grading tie-in** (創新性 / 困難度 / 完整度 / 報告書清晰度):

- *創新性 (innovation):* frame it honestly — your novelty is the **controlled,
  parameter-matched ablation + multi-seed + SB3 cross-check**, not "inventing
  attention". Reviewers reward honesty and rigour over inflated claims.
- *困難度 (difficulty):* the **from-scratch PPO/DQN math** and the
  **reproducibility pipeline** are your difficulty story — put the GAE/clipped
  objective equations front and centre in §IV.
- *完整度 (completeness):* multi-seed mean±std, an objective metrics table, a
  baseline cross-check, and a limitations section.
- *報告書清晰度 (clarity):* one figure + two tables that a reader can understand
  from the caption alone.

---

## 4. Figures and tables (the part students get wrong)

- **Reference every float in the text**: "as shown in Fig. 1", "Table III
  reports…". A figure nobody points to looks like filler.
- **Captions are self-contained.** A reader should understand
  Fig. 1 from its caption without reading the body.
- **Figure placement:** in LaTeX use `\begin{figure}[t]` (single column) or
  `\begin{figure*}[t]` (span both columns — good for your wide learning-curve
  plot). `[t]` = top of page.
- **Export figures at high resolution.** Your plots are already saved at
  150 dpi; for print, regenerate at **300 dpi** (`plt.savefig(..., dpi=300)`)
  and embed the PNG/PDF. Vector PDF is sharpest.
- **Tables use `booktabs`** (`\toprule`, `\midrule`, `\bottomrule`) — no
  vertical lines. Caption **above** the table, label "TABLE I" in small caps.

Example (a wide figure + a table):

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{comparison_aggressive.png}
  \caption{Learning curves (mean $\pm$ std over 3 seeds) for the three core
           configurations on \texttt{highway-v0}, aggressive style.}
  \label{fig:curves}
\end{figure*}

\begin{table}[t]
  \centering
  \caption{Objective evaluation (100 episodes, common base config).}
  \label{tab:eval}
  \begin{tabular}{lcccc}
    \toprule
    Model & Crash \% & Survival & Speed & Lane chg. \\
    \midrule
    PPO+Attn & ... & ... & ... & ... \\
    PPO+MLP  & ... & ... & ... & ... \\
    DQN+MLP  & ... & ... & ... & ... \\
    \bottomrule
  \end{tabular}
\end{table}
```

---

## 5. Markdown → LaTeX cheat-sheet

| In `report.md` (Markdown) | In LaTeX (IEEEtran) |
|---|---|
| `## I. Introduction` | `\section{Introduction}` (number is automatic) |
| `### A. PPO` | `\subsection{PPO}` |
| `**bold**` | `\textbf{bold}` |
| `*italic*` | `\textit{italic}` |
| `` `code` `` / `highway-v0` | `\texttt{highway-v0}` |
| fenced ``` block of equations | `\begin{equation}...\end{equation}` or `align` |
| `[1]` citation | `\cite{schulman2017ppo}` |
| Markdown table | `tabular` + `booktabs` |
| `γ, λ, ε, ±` | `$\gamma,\lambda,\epsilon,\pm$` |
| image | `\includegraphics{...}` |

**Equations:** the GAE / clipped-surrogate blocks in `report.md` are written as
plain text so you can read them; in LaTeX wrap them properly, e.g.

```latex
\begin{equation}
L^{\text{CLIP}} = \mathbb{E}\big[\min(\rho_t \hat{A}_t,\,
   \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)\big].
\end{equation}
```

---

## 6. References in IEEE style

- Cited **by number in order of appearance**: the first work you cite is `[1]`.
- In LaTeX, use a `.bib` file + `\bibliographystyle{IEEEtran}` and let it sort,
  or use the manual `thebibliography` environment.
- Format: `[1] A. Author and B. Author, "Title," Venue, year.`
- `report.md` already lists the 7 references in IEEE style; reuse them. A
  starter BibTeX entry:

```bibtex
@article{schulman2017ppo,
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla
          and Radford, Alec and Klimov, Oleg},
  title={Proximal Policy Optimization Algorithms},
  journal={arXiv preprint arXiv:1707.06347}, year={2017}}
```

---

## 7. Hitting 7–8 pages without padding

Two IEEE columns hold a lot. With one wide figure, two tables, and the equations,
your content should land near 7 pages. If you are **short**:

- Add a **small architecture diagram** (boxes for projection → attention →
  heads). Diagrams legitimately fill space and aid clarity.
- Add the **reward-shaping / style** comparison figure (conservative vs
  aggressive behaviour).
- Expand Discussion with a per-metric paragraph.

If you are **over 8 pages**: shrink the equation spacing, move hyperparameter
lists into a compact table, and cut repeated sentences. Do **not** shrink the
font below 10 pt.

---

## 8. Keeping the "AI 撰寫比例" low (course rule)

The course says AI-written proportion should not be too high. Practical, honest
ways to comply — and genuinely improve the report:

1. **Rewrite each paragraph in your own words** after reading it. You ran these
   experiments; narrate them as *your* decisions ("I matched the parameter
   counts so that…").
2. **Insert first-person engineering notes** the AI cannot know: bugs you hit,
   why you picked `duration=80`, how long training took on *your* laptop, what
   surprised you (e.g. that the three methods nearly tie on return but differ on
   crash rate).
3. **Hand-draw or hand-build the architecture figure.**
4. **Add a screenshot or frame** from `highway-env` running your policy.
5. Keep `report.md` as your **structured outline / fact-sheet**, then write the
   final prose yourself. That inverts the ratio in your favour and is exactly
   how the report becomes defensible in a viva.

> Treat `report.md` as a rigorously-checked skeleton with correct numbers and
> equations — not as the final text to paste verbatim.

---

## 9. Checklist before you submit

- [ ] Two-column IEEEtran, 10 pt, **7–8 pages** incl. references.
- [ ] Title, author block, abstract (with **one headline number**), index terms.
- [ ] Sections I–IX present; every figure/table is referenced in the text.
- [ ] Fig. 1 caption is self-contained; Table I/II/III formatted with booktabs.
- [ ] Equations numbered; symbols defined on first use.
- [ ] References in IEEE numbered style, cited in order.
- [ ] You rewrote the prose in your own voice; added ≥1 figure the AI didn't make.
- [ ] Code + logs + models + compressed demo video zipped alongside the PDF.
- [ ] Demo video **re-encoded smaller** (e.g. `ffmpeg -i in.mp4 -vcodec libx264
      -crf 28 out.mp4`), not just zipped.

---

*Companion files:* [report.md](report.md) (the content, with correct numbers and
equations) and the project `README.md` (setup + how to reproduce).
