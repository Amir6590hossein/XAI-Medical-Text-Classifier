import json
import requests
import os
import google.generativeai as genai

try:
    from groq import Groq
except ImportError:
    Groq = None

class Orchestrator:
    def __init__(self, max_trials=10, provider="Perplexity", api_key=None, model_name=None):
        self.max_trials = max_trials
        self.current_trial = 0
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        
        self.client = None
        
        if self.api_key:
            if provider == "Google Gemini":
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model_name)
                
            elif provider == "Groq":
                if Groq:
                    self.client = Groq(api_key=api_key)
                else:
                    print("Warning: Groq library is not installed.")
            

    def plan_next_trial(self, memory):
        self.current_trial += 1
        
        if self.current_trial > self.max_trials:
            return None, "Budget Exhausted"

        history = memory.history
        
        if not self.api_key:
             return self._get_heuristic_plan(history)

        try:
            return self._query_llm(history)
        except Exception as e:
            print(f"[Orchestrator] LLM Error: {e}. Switching to Heuristic.")
            return self._get_heuristic_plan(history)

    # ============================================================
    # [NEW FEATURE] XAI Medical Explanation Generator
    # ============================================================
    def generate_explanation(self, text_snippet, predicted_class, keywords):
        """
        Generates a natural language explanation for the model's prediction
        based on extracted keywords (Integrated Gradients).
        """
        # Fallback if no API key is present
        if not self.api_key:
            return (f"AI Prediction based on key terms: {', '.join(keywords)}. "
                    "(Connect an LLM to generate a detailed medical clinical report).")

        # 1. Construct the Prompt
        system_prompt = (
            "You are an expert Pathologist and Medical AI Consultant. "
            "Your task is to explain a Deep Learning model's diagnosis to a clinician. "
            "Be concise, professional, and evidence-based."
        )
        
        user_message = (
            f"**Patient Case Snippet:** \"{text_snippet}\"\n"
            f"**AI Predicted Diagnosis:** Class {predicted_class}\n"
            f"**Key Clinical Drivers (Identified by Attention/Gradients):** {', '.join(keywords)}\n\n"
            "**Task:** Write a short paragraph (2-3 sentences) explaining medically WHY these specific keywords "
            "support this diagnosis. Do not mention 'neural networks' or 'math'; focus on the clinical symptoms and pathology."
        )

        explanation = "Explanation generation failed."

        try:
            # 2. Call the appropriate Provider
            
            # --- Google Gemini ---
            if self.provider == "Google Gemini":
                full_prompt = system_prompt + "\n\n" + user_message
                response = self.client.generate_content(full_prompt)
                explanation = response.text
                
            # --- Groq ---
            elif self.provider == "Groq":
                if not self.client: raise ImportError("Groq client not initialized")
                chat = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    temperature=0.3, # Low temperature for factual consistency
                )
                explanation = chat.choices[0].message.content

            # --- Perplexity ---
            elif self.provider == "Perplexity":
                url = "https://api.perplexity.ai/chat/completions"
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                }
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                explanation = response.json()['choices'][0]['message']['content']

        except Exception as e:
            print(f"[Orchestrator] Explanation Error: {e}")
            return f"Could not generate clinical explanation. (Error: {str(e)})"

        return explanation

    # ============================================================
    # Existing AutoML Logic
    # ============================================================
    def _query_llm(self, history):
        system_prompt = (
            "You are an expert AutoML Orchestrator for Medical Text Classification (NLP). "
            "Maximize Accuracy with limited budget. "
            "Available models: ['bert', 'biobert', 'roberta']. "
            "Strategies: ['head_only', 'lora', 'adapter', 'full_ft']. "
            "Return ONLY a valid JSON object with keys: 'model_name', 'strategy', 'batch_size', 'lr', 'epochs', 'reasoning'. "
            "No markdown, just raw JSON."
        )
        
        history_str = json.dumps(history, indent=2)
        user_message = f"History: {history_str}\nSuggest config for Trial {self.current_trial}."
        
        result_text = ""
        
        # 1. Google Gemini Logic
        if self.provider == "Google Gemini":
            full_prompt = system_prompt + "\n\n" + user_message
            response = self.client.generate_content(full_prompt)
            result_text = response.text
            
        # 2. Groq Logic
        elif self.provider == "Groq":
            if not self.client: raise ImportError("Groq client not initialized")
            chat = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=self.model_name,
                temperature=0.2,
            )
            result_text = chat.choices[0].message.content
            
        # 3. Perplexity Logic
        elif self.provider == "Perplexity":
            url = "https://api.perplexity.ai/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result_text = response.json()['choices'][0]['message']['content']

        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        try:
            config = json.loads(result_text)
        except json.JSONDecodeError:
            start_idx = result_text.find('{')
            end_idx = result_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = result_text[start_idx:end_idx+1]
                config = json.loads(json_str)
            else:
                raise ValueError("Could not parse JSON from LLM response")

        reasoning = config.pop('reasoning', 'AI generated strategy.')
        
        config['batch_size'] = int(config.get('batch_size', 16))
        config['epochs'] = int(config.get('epochs', 3))
        config['lr'] = float(config.get('lr', 2e-5))
        
        return config, reasoning

    def _get_heuristic_plan(self, history):
        if self.current_trial == 1:
            return {'model_name': 'bert', 'strategy': 'head_only', 'batch_size': 16, 'lr': 1e-3, 'epochs': 3}, "Baseline: BERT Head-Only"
        if self.current_trial == 2:
            return {'model_name': 'biobert', 'strategy': 'lora', 'batch_size': 16, 'lr': 2e-4, 'epochs': 3}, "Exploration: BioBERT with LoRA"
            
        if history:
            best_trial = sorted(history, key=lambda x: x['metrics'].get('accuracy', 0), reverse=True)[0]
            config = best_trial['config'].copy()
            config['epochs'] = min(config['epochs'] + 1, 5)
            config['lr'] = config['lr'] * 0.5
            return config, f"Refining Trial {best_trial['trial_id']}"
        else:
             return {'model_name': 'roberta', 'strategy': 'full_ft', 'batch_size': 8, 'lr': 2e-5, 'epochs': 3}, "Fallback Heuristic"