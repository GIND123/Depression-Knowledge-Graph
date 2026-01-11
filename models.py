import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login


class DepressionAnalysisPipeline:

    # ==========================================
    # 1. PUBLIC CONSTANTS 
    # ==========================================
    DEFAULT_FOLDERS = {
        "repo_id": "GOVINDFROM/DepressionPipeline",
        "hope": "xlmr_hope_malayalam",
        "sentiment": "xlmr_malayalam_sentiment",
        "phq": "xlmr_phq9_bucket_mock/checkpoint-150"
    }

    def __init__(self, folders=None):
        """
        Initialize the pipeline.
        Args:
            folders (dict, optional): Override default folder paths. 
                                      If None, uses DepressionAnalysisPipeline.DEFAULT_FOLDERS.
        """
        # 2. Setup Configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set instance folders (Defaults to the class constant if not provided)
        self.folders = folders if folders else self.DEFAULT_FOLDERS.copy()

        # 3. Authentication Logic (Try Token -> Fallback to Anonymous)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("hf_token")
        
        if hf_token:
            login(token=hf_token)
            print(f"✅ Authenticated with Token on {self.device}")
        else:
            # Login anonymously (useful for public repos)
            # This avoids the "RuntimeError" you had before
            print(f"No token found. Using Anonymous mode (Public Repo only).")

        # 4. Load Tokenizer
        print("Loading Tokenizer -->")
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)

        # 5. Load Models using the stored folder paths
        self.repo_id = self.folders["repo_id"]
        self.hope_mod = self._load_hf_model(self.folders["hope"])
        self.sent_mod = self._load_hf_model(self.folders["sentiment"])
        self.phq_mod  = self._load_hf_model(self.folders["phq"])

        # Verification
        if None in [self.repo_id, self.hope_mod, self.sent_mod, self.phq_mod]:
            raise RuntimeError("One or more models failed to load. Check Repo ID and Folders.")

        print(f"Pipeline Ready. Using models from: {self.repo_id}")

        # Keywords for override logic
        self.HOPELESS_KWS = [
            "മാറില്ല", "പ്രതീക്ഷയില്ല", "future ഇല്ല", "hopeless",
            "nothing will change", "ഒന്നും മാറില്ല", "ഇനി ഒന്നും മാറില്ല"
        ]

    def _load_hf_model(self, subfolder):
        """Private helper to load a single model from HF Hub."""
        print(f"Loading model from {subfolder}...")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.repo_id,
                subfolder=subfolder
            ).to(self.device).eval()
            return model
        except Exception as e:
            print(f"Error loading {subfolder}: {e}")
            return None

    def _predict_one(self, text, model):
        """Private helper to run inference on a single string."""
        # Handle empty text to prevent crashes
        if not text or not isinstance(text, str):
            text = ""

        x = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        x = {k: v.to(self.device) for k, v in x.items()}

        with torch.no_grad():
            out = model(**x)

        probs = out.logits.softmax(dim=-1)[0].detach().cpu().numpy()
        pred_id = int(np.argmax(probs))
        label = model.config.id2label.get(pred_id, str(pred_id))

        return label, float(probs[pred_id]), probs.tolist()

    def _distress_override(self, text, sent_label):
        """Applies keyword-based override rules."""
        t = (text or "").lower()
        if any(k.lower() in t for k in self.HOPELESS_KWS):
            return "Negative_override"
        return sent_label

    def analyze_session(self, patient_id, session_id, timestamp, turns):
        """
        Analyzes a list of turns and returns a comprehensive session summary.
        """
        turn_results = []

        # 1. Analyze individual turns
        for t in turns:
            hope_label, hope_conf, _ = self._predict_one(t, self.hope_mod)
            sent_label, sent_conf, _ = self._predict_one(t, self.sent_mod)
            sent_label = self._distress_override(t, sent_label)

            turn_results.append({
                "text": t,
                "hope": {"label": hope_label, "confidence": hope_conf},
                "sentiment": {"label": sent_label, "confidence": sent_conf},
            })

        # 2. Analyze full session (PHQ-9)
        session_text = "\n".join(turns)
        phq_label, phq_conf, phq_probs = self._predict_one(session_text, self.phq_mod)

        # 3. Calculate aggregates
        if not turn_results:
            hope_rate, distress_rate, distress_score = 0.0, 0.0, 0.0
        else:
            hope_rate = sum(r["hope"]["label"] == "Hope_speech" for r in turn_results) / len(turn_results)
            DISTRESS_LABELS = {"negative", "negative_override"}
            distress_rate = sum(r["sentiment"]["label"].lower() in DISTRESS_LABELS for r in turn_results) / len(turn_results)
            distress_score = round(0.7 * distress_rate + 0.3 * (1 - hope_rate), 2)

        # 4. Identify risky turns
        top_risky_turns = sorted(
            turn_results,
            key=lambda r: (
                r["sentiment"]["label"].lower() in {"negative", "negative_override"},
                r["sentiment"]["confidence"]
            ),
            reverse=True
        )[:3]

        return {
            "patient_id": patient_id,
            "session_id": session_id,
            "timestamp": timestamp,
            "turn_analysis": turn_results,
            "session_summary": {
                "hope_rate": round(hope_rate, 2),
                "distress_rate": round(distress_rate, 2),
                "distress_score": distress_score,
                "phq_bucket": phq_label,
                "phq_conf": round(phq_conf, 6),
                "phq_probs": [round(p, 6) for p in phq_probs],
                "top_risky_turns": top_risky_turns,
                "doctor_note": "Screening signal only; interpret clinically."
            }
        }

    def to_kg_json(self, session_out):
        """Converts the session analysis output into Knowledge Graph nodes/edges."""
        nodes, edges = [], []
        pid = session_out["patient_id"]
        sid = session_out["session_id"]

        nodes.append({"id": f"patient:{pid}", "type": "Patient", "patient_id": pid})
        nodes.append({"id": f"session:{sid}", "type": "Session", "session_id": sid, "timestamp": session_out["timestamp"]})
        edges.append({"source": f"patient:{pid}", "target": f"session:{sid}", "type": "HAS_SESSION"})

        ss = session_out["session_summary"]
        nodes.append({"id": f"signal:{sid}:phq", "type": "Signal", "signal_type": "phq_bucket",
                      "label": ss["phq_bucket"], "confidence": ss["phq_conf"]})
        edges.append({"source": f"session:{sid}", "target": f"signal:{sid}:phq", "type": "HAS_SIGNAL"})

        for i, tr in enumerate(session_out["turn_analysis"], start=1):
            tid = f"{sid}:t{i}"
            nodes.append({"id": f"turn:{tid}", "type": "Turn", "turn_id": tid, "index": i, "text": tr["text"]})
            edges.append({"source": f"session:{sid}", "target": f"turn:{tid}", "type": "HAS_TURN"})

            nodes.append({"id": f"signal:{tid}:hope", "type": "Signal", "signal_type": "hope",
                          "label": tr["hope"]["label"], "confidence": tr["hope"]["confidence"]})
            edges.append({"source": f"turn:{tid}", "target": f"signal:{tid}:hope", "type": "HAS_SIGNAL"})

            nodes.append({"id": f"signal:{tid}:sent", "type": "Signal", "signal_type": "sentiment",
                          "label": tr["sentiment"]["label"], "confidence": tr["sentiment"]["confidence"]})
            edges.append({"source": f"turn:{tid}", "target": f"signal:{tid}:sent", "type": "HAS_SIGNAL"})

        return {"nodes": nodes, "edges": edges}