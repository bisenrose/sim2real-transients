# Paragraph to add to the paper (Sect. 4.7, "Robustness of the DES result"), after the runs

Fill in the bracketed values from tables/ablation.md and results/verdicts_9.json, then add the
table (tables/ablation.tex) as a new appendix table with label tab:ablation.

> Removing the three survey-artefact features (light-curve time span and the two fitted reference
> times) from the classifier altogether does not change our conclusions (Table~\ref{tab:ablation};
> predictions recorded before the runs). On DES the hybrid rule still beats consistent
> preprocessing for [all four] models ([range]); on ZTF and YSE it remains below per-survey
> rescaling for [all three] tree models, and consistent preprocessing remains the best option on
> YSE. Without these features the domain classifier still separates PLAsTiCC from DES, ZTF and YSE
> with AUC [x], [y] and [z], so the near-absence of overlap that defeats density-ratio methods is
> not caused by the survey-design features alone.

If any prediction fails, say so plainly instead, e.g. "...except that on [target] the hybrid rule
no longer [...]", and send the results to Claude to reword the Discussion accordingly.
