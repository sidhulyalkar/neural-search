# LaTeX and Overleaf Fixes

This document addresses the warnings reported from Overleaf.

## Warning: `Command \showhyphens has changed`

This is likely from the interaction between modern LaTeX and packages such as `microtype`.

Recommended action:

- Keep `microtype` unless it breaks compilation.
- Treat this as low priority if the PDF renders correctly.
- If needed, switch Overleaf's TeX Live version or temporarily comment out `microtype` only for debugging.

```latex
\usepackage{microtype}
```

## Warning: `'h' float specifier changed to 'ht'`

LaTeX could not place a figure or table exactly “here.”

Replace:

```latex
\begin{table}[h]
```

with:

```latex
\begin{table}[t]
```

or:

```latex
\begin{table}[htbp]
```

For conference-style papers, `[t]` is usually cleaner.

## Overfull hbox around field lists

Long `\texttt{}` strings do not break well.

Problematic pattern:

```latex
Fields $\mathcal{F}$ include: \texttt{title}, \texttt{description}, \texttt{brain_regions}, \texttt{behavioral_events}, ...
```

Better:

```latex
Fields $\mathcal{F}$ include:
\begin{center}
\small
\texttt{title}, \texttt{description}, \texttt{tasks}, \texttt{modalities},
\texttt{species}, \texttt{brain\_regions}, \texttt{behavioral\_events},
and \texttt{data\_standards}.
\end{center}
```

Best: replace long inline lists with a compact table.

## Overfull hbox from long equations

Break long equations across lines.

Instead of:

```latex
\[
\mathcal{H} = (\mathcal{U}, \{\mathcal{D}_i\}_{i=1}^{k}), \quad \mathcal{U} \subset \mathbb{R}^{d_u}, \quad \mathcal{D}_i \subset \mathbb{R}^{d_i}
\]
```

Use:

```latex
\[
\mathcal{H} =
\left(
\mathcal{U},
\{\mathcal{D}_i\}_{i=1}^{k}
\right),
\quad
\mathcal{U} \subset \mathbb{R}^{d_u},
\quad
\mathcal{D}_i \subset \mathbb{R}^{d_i}.
\]
```

Or use `aligned`:

```latex
\[
\begin{aligned}
\mathcal{H} &=
\left(\mathcal{U}, \{\mathcal{D}_i\}_{i=1}^{k}\right), \\
\mathcal{U} &\subset \mathbb{R}^{d_u}, \\
\mathcal{D}_i &\subset \mathbb{R}^{d_i}.
\end{aligned}
\]
```

## Underfull hbox in appendix tables

Long table cells can create bad line breaks.

Add these packages:

```latex
\usepackage{tabularx}
\usepackage{array}
```

Replace:

```latex
\begin{tabular}{lp{6cm}}
```

with:

```latex
\begin{tabularx}{\linewidth}{l>{\raggedright\arraybackslash}X}
```

Use:

```latex
\begin{table}[t]
\centering
\small
\begin{tabularx}{\linewidth}{l>{\raggedright\arraybackslash}X}
\toprule
Field & Description \\
\midrule
\texttt{species} & Subject species extracted from dataset metadata, publication text, or schema-level evidence. \\
\texttt{modality} & Recording or imaging modality, such as Neuropixels, calcium imaging, EEG, MEG, fMRI, or behavior-only data. \\
\bottomrule
\end{tabularx}
\caption{Example metadata fields used by Neural Search.}
\end{table}
```

## More robust table columns

For wide tables, define reusable column types:

```latex
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
```

Then:

```latex
\begin{tabularx}{\linewidth}{L{0.25\linewidth}Y}
```

## Long URLs

Add:

```latex
\usepackage{xurl}
```

This allows URLs to break more gracefully.

## Long code-like identifiers

Use manual breaks if needed:

```latex
\texttt{behavioral\_\allowbreak events}
```

or avoid inline code style for long field names in paragraphs.

## Practical warning triage

Prioritize:

1. Compilation errors
2. Missing references
3. Figure/table overflow
4. Overfull boxes visible in final PDF
5. Underfull boxes that look ugly
6. Harmless package warnings

Do not spend hours chasing harmless warnings if the PDF is readable and submission-compliant.
