import os
import json
import torch
import torch.nn.functional as F
import argparse
import sys
import numpy as np
from transformers import AutoTokenizer

sys.path.append(os.getcwd())

try:
    from src.model_layer.model_factory import get_base_model
    from src.model_layer.tuners import apply_tuning_strategy
    from src.execution.evaluator import Evaluator
    # [Step 1] Import Captum for XAI
    from captum.attr import LayerIntegratedGradients
except ImportError as e:
    print(" Error importing modules. Make sure you are running this from the project root.")
    print(" Also ensure 'captum' is installed: pip install captum")
    raise e

# ==========================================
# 1. Single Model Inference (With Advanced XAI)
# ==========================================
class MedicalTextInference:
    def __init__(self, experiment_dir="experiments", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = experiment_dir
        
        print(f"[Inference] Searching for best model in {experiment_dir}...")
        self.best_config, self.trial_id = self._load_best_trial_info()
        
        self.tokenizer = None
        self.model = self._build_model()
        
        self._load_weights()
        
        print(f" Inference Engine Ready! (Loaded Trial {self.trial_id}: {self.best_config['model_name']})")

    def _load_best_trial_info(self):
        results_path = os.path.join(self.experiment_dir, "results.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found at {results_path}. Run main.py first.")
            
        with open(results_path, 'r') as f:
            history = json.load(f)

        if not history:
            raise ValueError("Experiment history is empty.")
            
        best_trial = sorted(history, key=lambda x: x['metrics']['accuracy'], reverse=True)[0]
        return best_trial['config'], best_trial['trial_id']

    def _build_model(self):
        model_name = self.best_config['model_name']
        strategy = self.best_config['strategy']
        
        print(f"[Inference] Rebuilding {model_name} with strategy: {strategy}...")
        
        model_mapping = {
            'bert': 'bert-base-uncased',
            'biobert': 'dmis-lab/biobert-v1.1',
            'roberta': 'roberta-base'
        }
        hf_name = model_mapping.get(model_name, model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
      
        n_classes = self.best_config.get('num_classes', 5) 
        model = get_base_model(model_name, num_classes=n_classes, pretrained=False)
        model = apply_tuning_strategy(model, strategy)
        
        return model.to(self.device)

    def _load_weights(self):
        weights_path = os.path.join(self.experiment_dir, f"model_trial_{self.trial_id}.pth")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
            
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval() 

    def _get_embedding_layer(self):
        """
        Dynamically find the embedding layer based on model architecture.
        Necessary for LayerIntegratedGradients.
        """
        if hasattr(self.model, 'bert'):
            return self.model.bert.embeddings
        elif hasattr(self.model, 'roberta'):
            return self.model.roberta.embeddings
        elif hasattr(self.model, 'distilbert'):
            return self.model.distilbert.embeddings
        else:
            # Fallback: Try to find the first module that looks like embeddings
            for name, module in self.model.named_modules():
                if 'embeddings' in name and isinstance(module, torch.nn.Module):
                    return module
            raise ValueError("Could not automatically find embedding layer for XAI.")

    def _forward_func(self, inputs, attention_mask=None):
        """
        Wrapper for Captum: returns logits directly
        """
        return self.model(inputs, attention_mask=attention_mask).logits

    # [Step 3 Logic] Aggregation & Normalization
    def _aggregate_tokens(self, tokens, attributions, threshold=0.3):
        """
        Merges sub-words (e.g. 'Hem', '##orr', '##hage') into one word ('Hemorrhage').
        Sums their scores and normalizes to [0, 1].
        """
        aggregated_words = []
        current_word = ""
        current_score = 0.0
        
        # 1. Merge Sub-words
        for token, score in zip(tokens, attributions):
            score_val = score.item()
            
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
                continue
                
            # Check for sub-word prefix (BERT uses '##')
            if token.startswith("##"):
                current_word += token.replace("##", "")
                current_score += score_val
            else:
                # If there was a previous word, save it
                if current_word:
                    aggregated_words.append((current_word, current_score))
                
                # Start new word
                current_word = token
                current_score = score_val
        
        # Append the last word
        if current_word:
            aggregated_words.append((current_word, current_score))
            
        # 2. Normalize Scores (Min-Max Scaling based on absolute max)
        if not aggregated_words:
            return []

        # Find max score to normalize
        max_score = max(abs(w[1]) for w in aggregated_words) if aggregated_words else 1.0
        if max_score == 0: max_score = 1.0

        normalized_words = []
        for word, score in aggregated_words:
            norm_score = score / max_score # Result is between -1 and 1
            
            if norm_score > threshold:
                normalized_words.append((word, round(norm_score, 4)))
        
        # Sort by importance
        normalized_words.sort(key=lambda x: x[1], reverse=True)
        
        return normalized_words

    def explain_prediction(self, input_ids, attention_mask, target_class_idx):
        """
        [Step 2] Performs Layer Integrated Gradients (IG).
        """
        embedding_layer = self._get_embedding_layer()
        lig = LayerIntegratedGradients(self._forward_func, embedding_layer)

        # Compute attributions
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=torch.zeros_like(input_ids),
            additional_forward_args=(attention_mask),
            target=target_class_idx,
            return_convergence_delta=True,
            internal_batch_size=1
        )

        # Aggregate along the embedding dimension
        attributions_sum = attributions.sum(dim=2).squeeze(0)
        
        # L2 Norm for raw vector (preprocessing step before token aggregation)
        attributions_norm = attributions_sum / torch.norm(attributions_sum)
        
        return attributions_norm

    def predict(self, text):
        if not text or not isinstance(text, str):
            return {"error": "Invalid input text"}

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
           
        except Exception as e:
            return {"error": f"Failed to tokenize text: {str(e)}"}

        # 1. Standard Prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            conf, pred_idx = torch.max(probs, 1)
            pred_idx = pred_idx.item()
            confidence = conf.item()

        predicted_class = pred_idx
        entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
        uncertainty_msg = Evaluator.get_uncertainty_statement(entropy)

        # 2. XAI Analysis (Explainability)
        key_words = []
        top_words_list = []
        
        try:
            # Calculate raw importance scores
            raw_attributions = self.explain_prediction(
                inputs['input_ids'], 
                inputs['attention_mask'], 
                predicted_class
            )
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # [Step 3] Aggregate & Filter
            # threshold=0.3 means we keep words with >30% relative importance
            key_words = self._aggregate_tokens(tokens, raw_attributions, threshold=0.3)
            
            # Create a simple list of top words for the Orchestrator later
            top_words_list = [w[0] for w in key_words[:5]]
            
        except Exception as e:
            print(f"[XAI Warning] Explanation failed: {e}")

        return {
            "text_snippet": text[:50] + "...",
            "predicted_class": predicted_class,
            "confidence_score": f"{confidence:.2%}",
            "entropy": f"{entropy:.4f}",
            "uncertainty_status": uncertainty_msg,
            "class_probabilities": probs.cpu().numpy().tolist()[0],
            "word_attributions": key_words,     # List of (word, score) for UI
            "top_influential_words": top_words_list # Clean list for LLM Prompt
        }

# ==========================================
# 2. Ensemble Inference
# ==========================================
class EnsembleInference:
    def __init__(self, experiment_dir="experiments", k=3, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_dir = experiment_dir
        self.models = []
        self.tokenizers = [] 
        
        results_path = os.path.join(self.experiment_dir, "results.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found at {results_path}")

        with open(results_path, 'r') as f:
            history = json.load(f)
        
        sorted_history = sorted(history, key=lambda x: x['metrics']['accuracy'], reverse=True)
        top_trials = sorted_history[:min(k, len(sorted_history))]
        
        print(f" [Ensemble] Initializing Ensemble with Top {len(top_trials)} models:")
        
        model_mapping = {
            'bert': 'bert-base-uncased',
            'biobert': 'dmis-lab/biobert-v1.1',
            'roberta': 'roberta-base'
        }

        for trial in top_trials:
            config = trial['config']
            trial_id = trial['trial_id']
            
            hf_name = model_mapping.get(config['model_name'], config['model_name'])
            tokenizer = AutoTokenizer.from_pretrained(hf_name)
            
            n_classes = config.get('num_classes', 5)
            model = get_base_model(config['model_name'], num_classes=n_classes, pretrained=False)
            model = apply_tuning_strategy(model, config['strategy'])
            
            weights_path = os.path.join(self.experiment_dir, f"model_trial_{trial_id}.pth")
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.eval()
                model.to(self.device)
                self.models.append(model)
                self.tokenizers.append(tokenizer)
            else:
                print(f" Warning: Weights for Trial {trial_id} not found. Skipping.")

        print("✅ Ensemble Engine Ready!")

    def predict(self, text):
        if not text:
            return {"error": "Empty text"}
            
        all_probs = []
        
        with torch.no_grad():
            for i, model in enumerate(self.models):
                tokenizer = self.tokenizers[i]
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                all_probs.append(probs)
        
        if not all_probs:
            return {"error": "No models loaded in ensemble."}

        avg_probs = torch.stack(all_probs).mean(dim=0)
        
        conf, pred_idx = torch.max(avg_probs, 1)
        predicted_class = pred_idx.item()
        confidence = conf.item()
        
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9)).item()
        uncertainty_msg = Evaluator.get_uncertainty_statement(entropy)

        return {
            "text_snippet": text[:50] + "...",
            "predicted_class": predicted_class,
            "confidence_score": f"{confidence:.2%}",
            "entropy": f"{entropy:.4f}",
            "uncertainty_status": uncertainty_msg,
            "class_probabilities": avg_probs.cpu().numpy().tolist()[0],
            "mode": "Ensemble (Soft Voting)"
        }

# ==========================================
# CLI (Command Line Interface)
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Text Inference Tool")
    parser.add_argument("--text", type=str, required=True, help="Input medical text to classify")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="Path to experiments directory")
    parser.add_argument("--ensemble", action="store_true", help="Use Ensemble (Soft Voting)")
    parser.add_argument("--k", type=int, default=3, help="Number of top models for ensemble")
    
    args = parser.parse_args()
    
    try:
        if args.ensemble:
            print(f" Starting Inference in ENSEMBLE Mode (Top {args.k})...")
            engine = EnsembleInference(experiment_dir=args.exp_dir, k=args.k)
        else:
            print(" Starting Inference in SINGLE BEST MODEL Mode (With XAI)...")
            engine = MedicalTextInference(experiment_dir=args.exp_dir)
        
        result = engine.predict(args.text)
        
        if "error" in result:
             print(f" Error: {result['error']}")
        else:
            print("\n" + "="*45)
            print(f" Input Snippet: {result['text_snippet']}")
            if "mode" in result:
                print(f" Mode: {result['mode']}")
            print("="*45)
            print(f" Predicted Class : {result['predicted_class']}")
            print(f" Confidence      : {result['confidence_score']}")
            print(f" Uncertainty     : {result['uncertainty_status']} (Entropy: {result['entropy']})")
            print("-" * 45)
            print("Probabilities per Class:")
            for i, p in enumerate(result['class_probabilities']):
                print(f"  Class {i}: {p:.2%}")
            
            # Show XAI Results if available
            if "word_attributions" in result:
                print("-" * 45)
                print("🧠 Key Drivers (XAI) - Normalized & Filtered:")
                for word, score in result['word_attributions']:
                    print(f"  • {word.ljust(15)} : {score:.4f}")
                    
            print("="*45 + "\n")
        
    except Exception as e:
        print(f" Critical Error: {e}")