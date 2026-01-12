
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

Natural language interactions are increasingly used in research settings to study mental health, wellbeing, and emotional trajectories over time. However, most existing computational approaches treat conversations as isolated text samples, producing static predictions that are difficult to interpret longitudinally. This limitation is particularly acute for low-resource languages, where culturally grounded expressions of distress are under-represented in standard benchmarks.

`MalaKG` is an open-source Python package for constructing and updating **dynamic, session-level knowledge graphs** from conversational text in Malayalam. The toolkit integrates multilingual transformer-based classifiers, rule-based linguistic signals, and graph-based aggregation to represent evolving indicators such as hope, distress, and symptom mentions across turns and sessions. By converting free-text dialogue into structured, temporally indexed graphs, `MalaKG` enables researchers to analyze conversational mental-health signals in a transparent and extensible manner without requiring access to proprietary platforms or large-scale clinical infrastructure.

# Statement of need

Research in computational mental health increasingly relies on conversational data from interviews, chat-based assessments, and digital interventions. While transformer-based classifiers can detect sentiment or risk at the utterance level, these outputs are often difficult to contextualize across time, sessions, or individuals. Existing tools typically return point estimates (e.g., sentiment labels or scores) without retaining the intermediate structure required for longitudinal analysis, auditability, or downstream integration with qualitative research workflows.

This gap is especially relevant for **low-resource and non-English languages**, where culturally specific expressions of distress may not align cleanly with dominant annotation schemes. `MalaKG` addresses this need by providing a reusable software framework that (i) incrementally updates a knowledge graph after each conversational turn, (ii) preserves turn-level evidence alongside session-level summaries, and (iii) exposes intermediate representations that can be inspected, visualized, or exported for further analysis. The target users include researchers in computational social science, digital mental-health research, human–computer interaction, and NLP practitioners working with under-represented languages.

# State of the field

A number of libraries support sentiment analysis, emotion classification, or mental-health risk prediction from text, often through pre-trained neural models. Separately, general-purpose graph libraries enable the construction of static knowledge graphs from structured inputs. However, existing tools rarely integrate **conversational NLP, temporal aggregation, and knowledge-graph construction** within a single, reusable research software package.

Moreover, many mental-health NLP pipelines are released as notebooks or model checkpoints, limiting reproducibility and extension. `MalaKG` adopts a different design philosophy: rather than proposing a new predictive model, it focuses on **software infrastructure** that links model outputs, linguistic rules, and temporal structure into a coherent graph representation. This design choice supports comparative research (e.g., swapping classifiers or rules), facilitates error analysis, and lowers the barrier for studying longitudinal conversational patterns in low-resource settings.

# Software design

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

