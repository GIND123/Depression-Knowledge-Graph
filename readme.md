
# Depression-Knowledge-Graph

A real-time AI agent chat that analyzes patient responses to track mental health metrics. It combines three NLP pipelines (Hope, Sentiment, PHQ-9) with a **Dynamic Knowledge Graph** that evolves with every conversation turn, providing a visual dashboard of the patient's changing state.

## Features

- **Multi-Model NLP Pipeline**:
    
    - **Hope Detection**: Identifies presence of optimism or hopelessness.
        
    - **Sentiment Analysis**: Tracks emotional tone (Positive/Negative/Mixed).
        
    - **PHQ-9 Estimation**: Predicts depression severity buckets (0-4, 5-9, etc.) from text.
        
- **Dynamic Knowledge Graph (KG)**:
    
    - Maintains a temporal graph of `Patient` $\to$ `Session` $\to$ `Summary` relationships.
        
    - Updates a **"Current State"** node in real-time, tracking current risk and average PHQ scores across sessions.
        
- **Heuristic Risk Assessment**:
    
    - Calculates a composite `Risk Level` (Low, Medium, High) by combining PHQ-9 severity, sentiment, and hope signals.
        
- **Live Dashboard Generation**:
    
    - Automatically generates a visual dashboard (`dashboard_step_X.png`) after every turn, combining the KG topology with a line plot of PHQ-9 trends.
        

## Architecture

### 1. `DepressionAnalysisPipeline`

This class manages the Hugging Face models and inference logic.

- **Models Used**:
    
    - `xlmr_hope_malayalam`
        
    - `xlmr_malayalam_sentiment`
        
    - `xlmr_phq9_bucket_mock`
        
- **HuggingFace Repo**: `GOVINDFROM/DepressionPipeline`
    

### 2. `DynamicKnowledgeGraph`

A wrapper around `NetworkX` that structures the analysis data into a directed graph.

- **Nodes**: `Patient`, `Session`, `SessionSummary`, `CurrentRisk`, `AvgPHQ`.
    
- **Edges**: `HAS_SESSION`, `HAS_SUMMARY`, `CURRENT_STATE`.
    

## Installation

1. **Clone the repository**
    
    Bash
    
    ```
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    
2. **Install dependencies**
    
    Bash
    
    ```
    pip install numpy torch transformers huggingface_hub networkx matplotlib python-dotenv
    ```
    

## Usage

You can run the script in two ways : 

### Scenario 1: Run Interactive Mode
	```python
	from model3 import run_chatbot
	
	run_chatbot(interactive=True)
	```

### Scenario 2: Run Auto Mode with Custom Sentences
	```python
	from model3 import run_chatbot
	my_custom_list = [
	    "എനിക്ക് ഒന്നിലും താൽപ്പര്യമില്ല; ഒന്നും സന്തോഷം തരുന്നില്ല.",
	    "ഉറക്കം ശരിയല്ല; രാത്രി മുഴുവൻ ഉണർന്നിരിക്കുന്നു.",
	    "ഭക്ഷണം കഴിക്കാൻ താൽപ്പര്യം കുറവാണ്; വിശപ്പ് തന്നെ മാറി.",
	    "എനിക്ക് കുറ്റബോധം തോന്നുന്നു; ഞാൻ എല്ലാവർക്കും ഒരു ഭാരം പോലെ               ആണെന്ന് തോന്നുന്നു.",
	    "ഇനി ഒന്നും മാറില്ലെന്ന് തോന്നുന്നു; പ്രതീക്ഷയില്ല."
	]
	
	run_chatbot(interactive=False, examples=my_custom_list)
	```
    
**Interactive Mode**:
  - The script will prompt: `Turn 1 | You:`.  
  - User can input a sentence.
  - The system will process the input, update the graph, and generate a dashboard.    
  - Type `exi` to exit the loop.
        
3. **Outputs**:
    
    - **Console Report**: Prints JSON summaries of risk, sentiment, and scores.
        
    - **`knowledge_graph.json`**: The full graph structure saved to disk.
        
    - **`dashboard_step_X.png`**: A visualization image created at each step.
        

## Dashboard Visualization

The system generates a `dashboard_step_X.png` after every interaction, containing two subplots:

1. **Knowledge Graph**: A visual representation of the patient node connecting to sessions and current state nodes.
    
2. **PHQ-9 Trend**: A line chart showing how the predicted depression severity has changed over the conversation turns.
    

## Requirements

- **Internet Connection**: Required to download models from Hugging Face on the first run.


