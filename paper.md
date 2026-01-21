
---
title: "MalaKG: A Python toolkit for longitudinal knowledge-graph construction from Malayalam mental-health conversations"
tags:
  - Python
  - natural language processing
  - mental health
  - knowledge graphs
  - low-resource languages
  - computational social science
authors:
  - name: Govind Arun
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Ashish Abraham
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Anna Thomas
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - name: Ajo Babu George
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 4
  - name: Aditya Mohanty
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 5
affiliations:
  - index: 1
    name: Independent Researcher, India
  - index: 2
    name: Independent Researcher, United States
  - index: 3
    name: Independent Researcher, India
date: 11 January 2026
bibliography: paper.bib
---

# Summary

Regional language mental health screening often relies on episodic questionnaires, failing to capture critical within-person symptom trajectories over time. We present a dynamic knowledge graph system designed for the longitudinal modeling of depression-related signals in Malayalam and Malayalam-English code-mixed conversations. User interactions are processed in real-time by three XLM-RoBERTa-base classifiers targeting hope speech, affective sentiment, and session-level PHQ-9 severity. To ensure safety and reliability, outputs are fused using a weighted distress score safeguarded by deterministic lexical overrides for explicit hopelessness. All entities and signals are persisted in a Neo4j property graph, creating an updatable substrate for temporal querying. We evaluate the system’s performance on standard datasets, achieving 86.7% accuracy for hope speech detection and 78.0% for PHQ-9 severity estimation. Results demonstrate that dynamic graph-based persistence enables interpretable, continuous risk monitoring, supporting clinician-in-the-loop triage and retrospective trajectory analysis.

# Statement of need

Research in computational mental health increasingly relies on conversational data from interviews, chat-based assessments, and digital interventions. While transformer-based classifiers can detect sentiment or risk at the utterance level, these outputs are often difficult to contextualize across time, sessions, or individuals. Existing tools typically return point estimates (e.g., sentiment labels or scores) without retaining the intermediate structure required for longitudinal analysis, auditability, or downstream integration with qualitative research workflows.

This gap is especially relevant for **low-resource and non-English languages**, where culturally specific expressions of distress may not align cleanly with dominant annotation schemes. `MalaKG` addresses this need by providing a reusable software framework that (i) incrementally updates a knowledge graph after each conversational turn, (ii) preserves turn-level evidence alongside session-level summaries, and (iii) exposes intermediate representations that can be inspected, visualized, or exported for further analysis. The target users include researchers in computational social science, digital mental-health research, human–computer interaction, and NLP practitioners working with under-represented languages.

# Related Works

While the integration of Knowledge Graphs (KGs) with Large Language Models is established in medical AI, current methodologies predominantly treat KGs as static references rather than dynamic, evolving structures. Standard approaches, such as those combining RLHF with KG guidance \cite{Wangref1} or utilizing GraphRAG for symptom verification \cite{Guoref2}, rely on fixed standards like DSM-5 without real-time structural evolution. Extraction-focused methods further limit adaptability; MedKG \cite{Linref3} and Text-to-GraphQL systems \cite{Niref4} depend on rigid schemas and pre-constructed repositories that fail to capture the nuances of ongoing dialogue. Most temporal systems also present challenges for personalized care: while some autonomous agents track the evolution of scientific literature \cite{zhangref5}, they ignore real-time user states. Consequently, existing diagnostic frameworks \cite{yuan6} remain tethered to general clinical rules, lacking the dynamic memory required to interpret unique, unfolding patient contexts.

# Implementation

`MalaKG` is implemented as a modular Python package with a clear separation between inference, graph construction, and analysis layers. Transformer-based text classifiers (e.g., XLM-RoBERTa models fine-tuned for sentiment, hope-related speech, or risk buckets) are wrapped behind a consistent prediction interface. Optional rule-based components capture language-specific symptom cues that may be missed by statistical models.

Each conversational turn is processed independently and then merged into a session-level knowledge graph, where nodes represent entities such as turns, symptoms, or inferred states, and edges encode temporal order and evidential relationships. The graph is updated incrementally as new turns arrive, allowing real-time or batch-style analysis using the same abstractions. The package supports export to standard formats (JSON and GraphML) to enable visualization and downstream processing with external tools.

Design trade-offs prioritize transparency and extensibility over end-to-end automation. Rather than hiding intermediate steps, `MalaKG` exposes model confidences, rule hits, and aggregation logic, allowing researchers to audit and adapt the pipeline to new domains, languages, or ethical constraints.

# Research impact statement

`MalaKG` is intended as enabling infrastructure for research on conversational mental-health analysis in low-resource languages. The package provides reproducible examples, documented APIs, and end-to-end scripts that demonstrate how dynamic knowledge graphs can be derived from real conversational data. While the software is at an early stage of public release, it is designed to support comparative experiments, methodological studies, and interdisciplinary collaborations that combine qualitative and quantitative analysis.

By releasing `MalaKG` as open-source research software, we aim to facilitate transparent experimentation, encourage reuse beyond a single dataset or study, and lower the barrier for researchers interested in longitudinal and interpretable analyses of conversational mental-health signals.

# AI usage disclosure

Generative AI tools were used during development to assist with code refactoring, documentation drafting, and example generation. All AI-assisted outputs were reviewed, tested, and validated by the authors, and all architectural and design decisions were made by the human contributors.

# Acknowledgements

The authors acknowledge discussions with colleagues in computational social science and mental-health research that informed the design goals of this software. No external funding was received for this work.

# References

