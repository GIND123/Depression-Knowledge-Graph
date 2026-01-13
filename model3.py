import os
import time
import json
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==========================================
# PIPELINE CLASS
# ==========================================
class DepressionAnalysisPipeline:
    DEFAULT_FOLDERS = {
        "repo_id": "GOVINDFROM/DepressionPipeline",
        "hope": "xlmr_hope_malayalam_v2",
        "sentiment": "xlmr_malayalam_sentiment_v2",
        "phq": "xlmr_phq9_bucket_mock_v2/checkpoint-250"
    }
    DISTRESS_LABELS = {"negative", "negative_override"}

    def __init__(self, folders=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.folders = folders if folders else self.DEFAULT_FOLDERS.copy()

        # Auth
        hf_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        if hf_token:
            login(token=hf_token)
        else:
            print(f"No token found. Using Anonymous mode.")

        print("Loading Tokenizer & Models...")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
        self.repo_id = self.folders["repo_id"]
        self.hope_mod = self._load_hf_model(self.folders["hope"])
        self.sent_mod = self._load_hf_model(self.folders["sentiment"])
        self.phq_mod  = self._load_hf_model(self.folders["phq"])

        self.HOPELESS_KWS = ["മാറില്ല", "പ്രതീക്ഷയില്ല", "future ഇല്ല", "hopeless", "nothing will change", "ഒന്നും മാറില്ല"]
        self.SYMPTOM_RULES = {
            "sleep_problem": ["ഉറക്കം", "നിദ്ര", "sleep"],
            "anhedonia":     ["താൽപ്പര്യമില്ല", "ഒന്നിലും ഇഷ്ടമില്ല", "no interest"],
            "hopelessness":  ["മാറില്ല", "പ്രതീക്ഷയില്ല", "hopeless", "future ഇല്ല"],
            "fatigue":       ["തളർച്ച", "tired", "energy ഇല്ല"],
            "appetite_change": ["ഭക്ഷണം", "appetite", "ഭക്ഷണം താൽപ്പര്യം കുറവ്"],
            "concentration": ["ശ്രദ്ധ", "focus", "concentration"],
            "guilt":         ["കുറ്റബോധം", "guilt", "burden"],
            "anxiety":       ["ആശങ്ക", "panic", "anxiety", "ഭയം"],
        }

    def _load_hf_model(self, subfolder):
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                self.repo_id, subfolder=subfolder
            ).to(self.device).eval()
        except Exception as e:
            print(f"Error loading {subfolder}: {e}")
            return None

    def _predict_one(self, text, model):
        if not text: text = ""
        x = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        x = {k: v.to(self.device) for k, v in x.items()}
        with torch.no_grad():
            out = model(**x)
        probs = out.logits.softmax(dim=-1)[0].detach().cpu().numpy()
        pred_id = int(np.argmax(probs))
        label = model.config.id2label.get(pred_id, str(pred_id))
        return label, float(probs[pred_id]), probs.tolist()

    def _extract_symptoms(self, text):
        t = (text or "").lower()
        hits = []
        for sym, kws in self.SYMPTOM_RULES.items():
            if any(k.lower() in t for k in kws):
                hits.append(sym)
        return hits

    def analyze_session(self, patient_id, session_id, timestamp, turns):
        turn_analysis = []
        for t in turns:
            hope_label, hope_conf, hope_probs = self._predict_one(t, self.hope_mod)
            sent_label, sent_conf, sent_probs = self._predict_one(t, self.sent_mod)
            
            rule_hit = False
            if any(k in t for k in self.HOPELESS_KWS):
                sent_label = "negative_override"
                rule_hit = True

            turn_analysis.append({
                "text": t,
                "hope": {"label": hope_label, "confidence": hope_conf},
                "sentiment": {"label": sent_label, "confidence": sent_conf},
                "symptoms": self._extract_symptoms(t),
                "rule_hit": rule_hit,
                "_debug": {"hope_probs": hope_probs, "sent_probs": sent_probs}
            })

        session_text = "\n".join(turns)
        phq_label, phq_conf, phq_probs = self._predict_one(session_text, self.phq_mod)

        if not turn_analysis:
            hope_rate, distress_rate, distress_score = 0.0, 0.0, 0.0
        else:
            hope_rate = sum(r["hope"]["label"] == "Hope_speech" for r in turn_analysis) / len(turn_analysis)
            distress_rate = sum(r["sentiment"]["label"].lower() in self.DISTRESS_LABELS for r in turn_analysis) / len(turn_analysis)
            distress_score = round(0.7 * distress_rate + 0.3 * (1 - hope_rate), 4)

        top_risky = sorted(
            turn_analysis, 
            key=lambda r: (r["sentiment"]["label"].lower() in self.DISTRESS_LABELS, r["sentiment"]["confidence"]), 
            reverse=True
        )[:3]

        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "turn_analysis": turn_analysis,
            "session_summary": {
                "hope_rate": round(hope_rate, 2),
                "distress_rate": round(distress_rate, 2),
                "distress_score": distress_score,
                "phq_bucket": phq_label,
                "phq_conf": round(phq_conf, 6),
                "phq_probs": phq_probs,
                "top_risky_turns": top_risky
            }
        }

    def print_live_report(self, payload):
        ss = payload["session_summary"]
        last = payload["turn_analysis"][-1]
        print("\n" + "="*70)
        print("Latest turn text:")
        print(last["text"])
        print("\nTurn-level predictions:")
        print(f"  Hope      : {last['hope']['label']} (conf={last['hope']['confidence']:.3f})")
        print(f"  Sentiment : {last['sentiment']['label']} (conf={last['sentiment']['confidence']:.3f})")
        
        symptoms = last.get("symptoms", [])
        print(f"  Symptoms  : {', '.join(symptoms) if symptoms else 'None'}")
        print(f"  Rule hit  : {'hopeless_override' if last.get('rule_hit') else 'None'}")

        print("\n Session summary:")
        print(f"  distress_score : {ss['distress_score']:.2f}")
        print(f"  phq_bucket     : {ss['phq_bucket']}")
        print("="*70)


# ==========================================
# KNOWLEDGE GRAPH CLASS
# ==========================================
class KnowledgeGraphManager:
    def __init__(self, json_path=None, graphml_path=None):
        self.g = nx.MultiDiGraph()
        self.json_path = json_path or "kg_store/kg.json"
        self.graphml_path = graphml_path or "kg_store/kg.graphml"

        if os.path.isfile(self.json_path):
            self.load_json(self.json_path)

    def upsert_node(self, node_id, ntype, **props):
        if node_id not in self.g:
            self.g.add_node(node_id, type=ntype, **props)
        else:
            self.g.nodes[node_id].update(props)

    def add_edge(self, src, dst, etype, **props):
        self.g.add_edge(src, dst, type=etype, **props)

    def update_from_session(self, payload):
        pid = payload["patient_id"]
        sid = payload["session_id"]
        ts  = payload["timestamp"]
        ss  = payload["session_summary"]
        turns = payload["turn_analysis"]

        patient_n = f"patient:{pid}"
        session_n = f"session:{sid}"

        self.upsert_node(patient_n, "Patient", patient_id=pid)
        self.upsert_node(session_n, "Session", session_id=sid, timestamp=ts, **ss)
        self.add_edge(patient_n, session_n, "HAS_SESSION")

        phq_sig = f"signal:{sid}:phq"
        self.upsert_node(phq_sig, "Signal", signal_type="phq_bucket", label=ss["phq_bucket"], confidence=ss["phq_conf"])
        self.add_edge(session_n, phq_sig, "HAS_SIGNAL")

        for i, tr in enumerate(turns, start=1):
            turn_n = f"turn:{sid}:t{i}"
            self.upsert_node(turn_n, "Turn", turn_id=f"{sid}:t{i}", index=i, text=tr["text"])
            self.add_edge(session_n, turn_n, "HAS_TURN")

            for signal_type in ["hope", "sentiment"]:
                sig_data = tr[signal_type]
                sig_id = f"signal:{sid}:t{i}:{signal_type}"
                self.upsert_node(sig_id, "Signal", signal_type=signal_type, **sig_data)
                self.add_edge(turn_n, sig_id, "HAS_SIGNAL")

            for sym in tr.get("symptoms", []):
                sym_n = f"symptom:{sym}"
                self.upsert_node(sym_n, "Symptom", name=sym)
                self.add_edge(turn_n, sym_n, "MENTIONS")

    def save(self):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        nodes = [{"id": n, **self.g.nodes[n]} for n in self.g.nodes()]
        edges = [{"source": u, "target": v, **data} for u, v, _, data in self.g.edges(keys=True, data=True)]
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

        g_clean = self.g.copy()
        for n, data in g_clean.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, (list, dict)): data[k] = str(v)
        for u, v, k, data in g_clean.edges(keys=True, data=True):
            for key, val in data.items():
                if isinstance(val, (list, dict)): data[key] = str(val)
        nx.write_graphml(g_clean, self.graphml_path)
        return self.json_path, self.graphml_path
    
    def load_json(self, in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.g.clear()
        for n in data["nodes"]:
            nid = n.pop("id")
            self.g.add_node(nid, **n)
        for e in data["edges"]:
            src = e.pop("source")
            dst = e.pop("target")
            self.g.add_edge(src, dst, **e)


# ==========================================
# DASHBOARD GENERATOR
# ==========================================
def generate_dashboard(pipeline, kg_manager, payload, step_num, show_plot=True):
    """
    Generates a 4-panel dashboard:
    [Distress Trend] [Knowledge Graph]
    [Hope Probs]     [PHQ-9 Buckets]
    """
    
    # Create save folder
    history_dir = "kg_store/history"
    os.makedirs(history_dir, exist_ok=True)
    save_path = f"{history_dir}/turn_{step_num}.png"

    # --- Data Prep ---
    ss = payload["session_summary"]
    ta = payload["turn_analysis"]
    
    # Trend Data
    running_scores = []
    DISTRESS_LABELS = {"negative", "negative_override"}
    for i in range(1, len(ta) + 1):
        subset = ta[:i]
        hr = sum(1 for x in subset if x["hope"]["label"] == "Hope_speech") / i
        dr = sum(1 for x in subset if x["sentiment"]["label"].lower() in DISTRESS_LABELS) / i
        score = 0.7 * dr + 0.3 * (1 - hr)
        running_scores.append(score)

    last_hope_probs = ta[-1]["_debug"]["hope_probs"][:2] if ta else [0, 0]
    phq_probs = ss["phq_probs"][:4]

    # --- PLOTTING ---
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)

    # 1. Top Left: Distress Score Trend
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, len(running_scores) + 1), running_scores, marker="o", color="tab:red", linewidth=2)
    ax1.set_title("Running Distress Score")
    ax1.set_xlabel("Turn Number")
    ax1.set_ylabel("Score (0-1)")
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)

    # 2. Top Right: Knowledge Graph
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f"Knowledge Graph (Turn {step_num})")
    
    if kg_manager.g.number_of_nodes() > 0:
        pos = nx.spring_layout(kg_manager.g, seed=42, k=0.6)
        node_colors = {"Patient": "#FF5733", "Session": "#33FF57", "Turn": "#3357FF", "Signal": "#FF33A8", "Symptom": "#F4D03F"}
        color_map = [node_colors.get(kg_manager.g.nodes[n].get("type"), "#CCCCCC") for n in kg_manager.g.nodes()]
        
        nx.draw_networkx_nodes(kg_manager.g, pos, ax=ax2, node_color=color_map, node_size=300, alpha=0.9)
        nx.draw_networkx_edges(kg_manager.g, pos, ax=ax2, alpha=0.4, edge_color="gray", arrows=True)
        
        # Smart labels (hide full text for cleaner view)
        labels = {}
        for node, data in kg_manager.g.nodes(data=True):
            if data.get("type") == "Symptom": labels[node] = data.get("name")
            elif data.get("type") == "Signal": labels[node] = data.get('label')[:10]
            else: labels[node] = node.split(":")[-1]
        nx.draw_networkx_labels(kg_manager.g, pos, ax=ax2, labels=labels, font_size=7)
    
    ax2.axis("off") # Hide axis for graph

    # 3. Bottom Left: Hope Probabilities
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(["No Hope", "Hope"], last_hope_probs, color=["tab:gray", "tab:green"])
    ax3.set_title("Latest Turn: Hope Analysis")
    ax3.set_ylim(0, 1.0)
    ax3.bar_label(bars, fmt='%.2f')

    # 4. Bottom Right: PHQ-9 Severity
    ax4 = fig.add_subplot(gs[1, 1])
    bars2 = ax4.bar(["None", "Mild", "Mod", "Sev"], phq_probs, color="tab:purple")
    ax4.set_title("Session-Level PHQ-9 Prediction")
    ax4.set_ylim(0, 1.0)
    ax4.bar_label(bars2, fmt='%.2f')

    plt.tight_layout()
    
    # Save first
    plt.savefig(save_path, dpi=150)
    print(f"Dashboard saved: {save_path}")

    # Show if interactive
    # if show_plot:
    #     plt.show()
    # else:
    plt.close()


# ==========================================
# MAIN CHATBOT LOGIC
# ==========================================
def run_chatbot(interactive=True, examples=None):
    
    # Default Examples
    DEFAULT_EXAMPLES = [
        "എനിക്ക് ഒന്നിലും താൽപ്പര്യമില്ല; ഒന്നും സന്തോഷം തരുന്നില്ല.",
        "ഉറക്കം ശരിയല്ല; രാത്രി മുഴുവൻ ഉണർന്നിരിക്കുന്നു.",
        "ഭക്ഷണം കഴിക്കാൻ താൽപ്പര്യം കുറവാണ്; വിശപ്പ് തന്നെ മാറി.",
        "എനിക്ക് കുറ്റബോധം തോന്നുന്നു; ഞാൻ എല്ലാവർക്കും ഒരു ഭാരം പോലെ ആണെന്ന് തോന്നുന്നു.",
        "ഇനി ഒന്നും മാറില്ലെന്ന് തോന്നുന്നു; പ്രതീക്ഷയില്ല."
    ]
    if examples is None: examples = DEFAULT_EXAMPLES

    # Init
    print("\n Initializing Pipeline & Knowledge Graph...")
    pipeline = DepressionAnalysisPipeline()
    kg = KnowledgeGraphManager(
        json_path="kg_store/kg_test.json", 
        graphml_path="kg_store/kg_test.graphml"
    )

    patient_id = "p_test_bot"
    session_id = f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp = datetime.now().isoformat()
    turns_history = []

    print(f"\n Chatbot Simulation Started | Mode: {'INTERACTIVE' if interactive else 'AUTOMATIC'}")
    print("----------------------------------------------------------------")

    step = 0
    while True:
        step += 1
        
        if interactive:
            try:
                user_text = input(f"\nTurn {step} | You: ").strip()
            except KeyboardInterrupt: break
            if user_text.lower() == "exi": break
            if not user_text: continue
        else:
            if step > len(examples): break
            user_text = examples[step-1]
            print(f"\nTurn {step} | User (Auto): {user_text}")
            time.sleep(1.0) 

        # --- Process ---
        turns_history.append(user_text)
        payload = pipeline.analyze_session(patient_id, session_id, timestamp, turns_history)
        kg.update_from_session(payload)
        kg.save()

        # --- Report ---
        pipeline.print_live_report(payload)
        
        # --- NEW: Generate Combined Dashboard (Plot + KG + Save) ---
        # If interactive: Show plot (User must close to continue)
        # If automatic: Don't show plot, just save
        generate_dashboard(pipeline, kg, payload, step_num=step, show_plot=interactive)

    print("\nSimulation Complete.")

if __name__ == "__main__":
    # Change to False to test automatic generation of images in /kg_store/history/

    run_chatbot(interactive=True)

