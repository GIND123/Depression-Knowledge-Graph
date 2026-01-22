
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

Mental health remains a major public health concern in Kerala despite the state’s strong indicators in healthcare access, with substantial levels of stress and depression reported across students and professionals \cite{brineshmental, swetha2025prevalence}. Large-scale syntheses reveal systematic disparities across demographic groups, yet current regional information systems rely heavily on cross-sectional surveys and static screening instruments \cite{rahna2024gender}. These episodic approaches fail to capture the evolving nature of symptom patterns over time, particularly in non-English contexts where help-seeking often occurs informally. This presents a critical research challenge: can we leverage conversational data to model longitudinal mental health trajectories in low-resource regional languages?

Our system leverages a dynamic Knowledge Graph (KG) framework to model mental health signals from Malayalam conversational text. While knowledge graph approaches have proven effective for medical reasoning and detecting community structures \cite{yu2020dynamic, ni2024knowledge, syam2025graph}, existing systems typically focus on static schemas or English-language contexts. A dynamic KG is particularly suitable for this domain as it supports the continuous integration of temporal signals and evolving patient states. Unlike static screening tools that provide isolated snapshots, a dynamic graph architecture naturally accommodates the fluidity of mental health symptoms, making it uniquely capable of structuring complex health information in regional settings.

To enable longitudinal modeling, the system incrementally integrates turn-level linguistic cues, affective signals, and session summaries into a structured graph representation. As new interactions occur, the graph evolves to explicitly track the emergence, persistence, and change of symptoms over time. By combining region-specific language processing with dynamic graph construction, the system advances beyond static assessments, providing a transparent and scalable framework for digital mental health monitoring that is sensitive to longitudinal variation.

# Related Works

While the integration of Knowledge Graphs (KGs) with Large Language Models is established in medical AI, current methodologies predominantly treat KGs as static references rather than dynamic, evolving structures. Standard approaches, such as those combining RLHF with KG guidance \cite{Wangref1} or utilizing GraphRAG for symptom verification \cite{Guoref2}, rely on fixed standards like DSM-5 without real-time structural evolution. Extraction-focused methods further limit adaptability; MedKG \cite{Linref3} and Text-to-GraphQL systems \cite{Niref4} depend on rigid schemas and pre-constructed repositories that fail to capture the nuances of ongoing dialogue. Most temporal systems also present challenges for personalized care: while some autonomous agents track the evolution of scientific literature \cite{zhangref5}, they ignore real-time user states. Consequently, existing diagnostic frameworks \cite{yuan6} remain tethered to general clinical rules, lacking the dynamic memory required to interpret unique, unfolding patient contexts.

# Implementation

`MalaKG` is implemented as a modular Python toolkit that integrates **transformer-based NLP classifiers**, **rule-based symptom extraction**, and **dynamic knowledge-graph management** to construct longitudinal representations of Malayalam and code-mixed mental-health conversations. The system is structured around three core components: the **analysis pipeline**, the **knowledge graph manager**, and the **dashboard generator**.

### 1. Analysis Pipeline

The `DepressionAnalysisPipeline` class orchestrates real-time processing of conversational turns. For each utterance, it performs:

- **Hope speech detection** using an XLM-RoBERTa-base classifier fine-tuned on Malayalam conversations.
- **Affective sentiment classification** via a second XLM-RoBERTa model, with deterministic overrides for explicit hopelessness phrases.
- **Session-level PHQ-9 severity estimation**, aggregating turn-level text into a single score for depression risk.

Predictions are represented as dictionaries containing labels, confidence scores, and optional probabilities for inspection. Additionally, a **symptom extraction module** identifies relevant mental-health cues such as sleep disturbances, anhedonia, fatigue, guilt, and anxiety using a curated set of lexical triggers.

Turn-level outputs are fused into a **weighted distress score**, combining negative sentiment and lack of hope to provide a real-time, interpretable measure of psychological distress. The pipeline exposes both turn-level and session-level outputs for downstream graph updates.

### 2. Knowledge Graph Management

The `KnowledgeGraphManager` class implements a **dynamic, updatable multi-directed graph** using NetworkX. Nodes represent patients, sessions, conversational turns, signals (e.g., hope, sentiment, PHQ-9), and detected symptoms, while edges encode temporal and relational structure:

- Patients → Sessions (`HAS_SESSION`)
- Sessions → Turns (`HAS_TURN`)
- Turns → Signals (`HAS_SIGNAL`)
- Turns → Symptoms (`MENTIONS`)

Graph updates occur incrementally with each new turn. Node properties store prediction metadata, including confidence scores and PHQ-9 buckets. Graph persistence is achieved in **JSON** and **GraphML** formats, enabling easy inspection, querying, and visualization.

### 3. Dashboard Generator

The toolkit provides an integrated visualization module, `generate_dashboard`, which produces a four-panel view per session:

1. **Running distress score trend**, showing temporal evolution across turns.
2. **Knowledge graph snapshot**, illustrating the current state of entities and relationships.
3. **Hope probability bar chart**, summarizing model output for the latest turn.
4. **PHQ-9 severity distribution**, visualizing session-level predictions.

Dashboards are saved automatically per turn and optionally displayed interactively to support clinician-in-the-loop evaluation or user-facing feedback.

### 4. Chatbot Simulation Interface

A lightweight interface wraps the pipeline and knowledge graph manager into a **turn-based simulation**. It supports interactive or scripted conversation flows and demonstrates the incremental building of the knowledge graph. At each step, the system:

1. Accepts a user utterance.
2. Generates turn-level predictions and updates the session summary.
3. Updates the knowledge graph with new nodes and edges.
4. Saves the graph and generates the dashboard visualization.

This design ensures that longitudinal conversational data are **preserved, auditable, and interpretable**, enabling both retrospective analysis and real-time monitoring.

### 5. Dependencies and Environment

The toolkit relies on standard Python scientific and NLP libraries, including `torch`, `transformers`, `networkx`, `numpy`, and `matplotlib`. Hugging Face Hub authentication is optional but allows seamless model loading. Environment variables are used to manage tokens and configuration paths.


# Research impact statement

`MalaKG` is intended as enabling infrastructure for research on conversational mental-health analysis in low-resource languages. The package provides reproducible examples, documented APIs, and end-to-end scripts that demonstrate how dynamic knowledge graphs can be derived from real conversational data. While the software is at an early stage of public release, it is designed to support comparative experiments, methodological studies, and interdisciplinary collaborations that combine qualitative and quantitative analysis.

By releasing `MalaKG` as open-source research software, we aim to facilitate transparent experimentation, encourage reuse beyond a single dataset or study, and lower the barrier for researchers interested in longitudinal and interpretable analyses of conversational mental-health signals.

# AI usage disclosure

Generative AI tools were used during development to assist with code refactoring, documentation drafting, and example generation. All AI-assisted outputs were reviewed, tested, and validated by the authors, and all architectural and design decisions were made by the human contributors.

# Acknowledgements

The authors acknowledge discussions with colleagues in computational social science and mental-health research that informed the design goals of this software. No external funding was received for this work.

# References

