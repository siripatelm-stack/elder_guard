import torch
from transformers import AutoModel, AutoTokenizer

from .model import CausalLSTM
from .text import MAX_TURNS, rule_flags, split_turns


class Detector:
    def __init__(self, embedder_dir, weights_path, threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(str(embedder_dir))
        self.bert = AutoModel.from_pretrained(str(embedder_dir)).to(self.device).eval()
        self.clf = CausalLSTM().to(self.device)
        self.clf.load_state_dict(torch.load(str(weights_path), map_location=self.device))
        self.clf.eval()

    @torch.no_grad()
    def embed(self, turns):
        enc = self.tokenizer(turns, padding=True, truncation=True, max_length=64,
                             return_tensors="pt").to(self.device)
        return self.bert(**enc).last_hidden_state[:, 0, :]      # [n_turns, 768]

    @torch.no_grad()
    def score(self, text):
        turns = split_turns(text)[:MAX_TURNS]
        if not turns:
            return {"turns": [], "alert_turn": None, "threshold": self.threshold}
        state, results, alert_turn = None, [], None
        for i, (t, e) in enumerate(zip(turns, self.embed(turns)), start=1):
            prob, state = self.clf.step(e.view(1, 1, -1), state)
            p = float(prob.item())
            flags = rule_flags(t)
            if alert_turn is None and (p >= self.threshold or flags):
                alert_turn = i                      # alert if the model OR a rule fires
            results.append({"turn": i, "text": t, "fraud_probability": round(p, 4), "flags": flags})
        return {"turns": results, "alert_turn": alert_turn, "threshold": self.threshold}