# Causal Hypothesis Generation

### From observational evidence to literature-informed research hypotheses

This project connects causal analysis, structured evidence statements, and large language models (LLMs) to develop hypotheses for further investigation.
The workflow makes each candidate traceable to its empirical motivation and evaluates its scientific quality and novelty relative to retrieved literature.

The framework has four stages: Causal Analysis of Observational Data, Translation of Causal Estimates, Candidate Hypothesis Generation, and Candidate Hypothesis Evaluation and Prioritization.
The empirical application examines recorded treatment completion in substance use treatment, comparing long-term with short-term residential care and intensive with non-intensive outpatient care.

**Release status:** This repository contains the October 11 replication code and prepared inputs for research use.
Saved Stage 1–4 result folders are not included in this snapshot.

[Workflow](#what-each-stage-contributes) · [Evaluation](#how-hypotheses-are-evaluated) · [Replication](#replication-package) · [Questions](#questions-and-contributions)

## What each stage contributes

| Stage | Method | Main output |
| :--- | :--- | :--- |
| **1. Causal Analysis of Observational Data** | Estimate treatment-effect heterogeneity, learn policy trees, and assess sensitivity to analytical choices. | Treatment-effect estimates, tree-defined subgroups, and diagnostics. |
| **2. Translation of Causal Estimates** | Extract complete tree paths and link subgroup conditions to saved estimates and evidence identifiers. | Structured evidence shared across hypothesis-generating models. |
| **3. Candidate Hypothesis Generation** | Prompt multiple models with the same aggregate evidence and check candidate structure and evidence references. | Hypotheses, proposed mechanisms, observable implications, and validity threats. |
| **4. Candidate Hypothesis Evaluation and Prioritization** | Assess scientific quality and relationships to prior studies, then apply deterministic eligibility rules and nondominated sorting. | Evaluator-specific ratings, novelty assessments, priority tiers, and comparisons. |

## How hypotheses are evaluated

Each evaluator assigns integer ratings from **1 to 5** on seven dimensions and explains its judgments.

| Dimension | Evaluation focus |
| :--- | :--- |
| Scientific validity | Whether the hypothesis is scientifically defensible and consistent with the supplied evidence. |
| Theoretical coherence | Whether the proposed mechanism has a coherent basis and distinguishes credible alternatives. |
| Scientific significance | The scientific importance of the proposed research question. |
| Feasibility | Whether the required data and study design can realistically be obtained and implemented. |
| Testability | Whether observable evidence can distinguish the hypothesis from plausible alternatives. |
| Clarity | Whether the population, treatment contrast, direction, and outcome are explicit. |
| Domain relevance (clinical relevance in this application) | Relevance to treatment decisions and administration in this empirical application. |

Domain relevance is operationalized as clinical relevance in this application.
Applications in other domains require an appropriate definition of practical relevance.

**Novelty** is assessed separately by examining the proposed relationship against retrieved studies.
The code derives a novelty rating and adjusts it for literature-coverage confidence.
Novelty therefore depends on the literature available to the evaluator.

**Prioritization uses no composite weights.**
Candidates must pass evidence-consistency checks and minimum eligibility thresholds.
Eligible candidates are assigned Pareto tiers across the quality ratings and coverage-adjusted novelty.
A candidate is in the first tier when no other eligible candidate is at least as highly rated on every dimension and more highly rated on at least one.
Tiers are computed separately for each evaluator.

Spider charts display the dimensions together.
Comparisons across evaluators show where judgments agree and where further review is useful.

## Interpreting the outputs

The outputs are exploratory research hypotheses for subsequent investigation.
Treatment-effect estimates depend on the observational design and its identification assumptions.
LLM ratings guide review; empirical validation requires an appropriate follow-up study.
Policy-tree splits motivate subgroup hypotheses, while claims about moderation require dedicated tests.

## Replication package

The replication package is organized as follows.

| Folder | Contents |
| :--- | :--- |
| `S1/` | Data preparation, causal analysis, and completion inference. |
| `S1/sensitivity/` | Sensitivity analyses and summary generation. |
| `S2/` | Translation of Causal Estimates. |
| `S3/` | Candidate Hypothesis Generation. |
| `S4/` | Candidate Hypothesis Evaluation and Prioritization, with code-generated exports. |
| `src/` | Shared Python analysis modules. |
| `configs/` | Study specifications and analysis settings. |

The execution order is **S1 → S2 → S3 → S4**.
The sensitivity analyses use the relevant saved analysis inputs and outputs.

The workflow requires a Python environment with the analysis dependencies, study inputs, and the configured model services.
Local LLM calls use Ollama.
The Stage 3 configuration also supports Claude through its command-line interface.
Stage 4 requires a frozen literature corpus and its accompanying run configuration.
Model outputs can vary across reruns.

Run the stage notebooks in order after configuring their inputs and model services.
The current Stage 4 default points to a saved literature corpus that is absent from this snapshot; it also requires the corresponding run configuration.
Web of Science exports are not distributed in this repository.
Use the search strings in `S4/keyword_to_wos.txt` with your own authorized Web of Science access and save the exports locally as `S4/WoS_residential_20260926.txt` and `S4/WoS_outpatient_20260926.txt`.
These local exports are excluded from Git; they do not replace the separate frozen-corpus requirement described above.
A complete rerun of this packaged snapshot has not yet been verified.

## Questions and contributions

Use [Issues](https://github.com/xinyuan-wei-xw/causal-hypothesis-generation/issues) for methodological questions, reproducibility problems, or proposed improvements.
For a reproducibility issue, include the notebook, software environment, error message, and relevant configuration, with credentials and personal data removed.

Please propose changes to the analysis or evaluation criteria in an issue before modifying them.
This keeps methodological decisions explicit and makes revisions easier to review.

## Citation and reuse

Manuscript citation details and licensing information will be added with the research release.
