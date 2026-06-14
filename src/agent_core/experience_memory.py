import os
import json
import pandas as pd

class ExperienceMemory:
    def __init__(self):
        self.history = []

    def add_experience(self, trial_id, config, metrics):
        entry = {
            'trial_id': trial_id,
            'config': config,
            'metrics': metrics
        }
        self.history.append(entry)
        print(f"[Memory] Trial {trial_id} recorded with Acc: {metrics.get('accuracy', 0):.4f}")

    def get_best_trial(self):
        if not self.history:
            return None
        sorted_history = sorted(self.history, key=lambda x: x['metrics']['accuracy'], reverse=True)
        return sorted_history[0]

    def get_top_k_trials(self, k=3):
        if not self.history:
            return []
        sorted_history = sorted(self.history, key=lambda x: x['metrics']['accuracy'], reverse=True)
        return sorted_history[:min(k, len(sorted_history))]

    def to_dataframe(self):
        data = []
        for h in self.history:
            row = {'trial_id': h['trial_id']}
            row.update(h['config'])     
            row.update(h['metrics'])   
            data.append(row)
        return pd.DataFrame(data)

    def save_memory(self, path='experiments/results.json'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=4)