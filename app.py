import streamlit as st
import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import re
from transformers import AutoTokenizer

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.agent_core.llm_orchestrator import Orchestrator
from src.agent_core.experience_memory import ExperienceMemory
from src.model_layer.model_factory import get_base_model
from src.model_layer.tuners import apply_tuning_strategy
from src.execution.trainer import ModelTrainer
from src.execution.budget_manager import BudgetManager
from src.execution.evaluator import Evaluator
from src.data_layer.dataset_loader import MedicalTextDataset
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from inference import MedicalTextInference

# LLM Providers Configuration
MODEL_OPTIONS = {
    "Perplexity": [
        "sonar-pro",
        "sonar",
        "sonar-reasoning-pro",
        "sonar-reasoning"
    ],
    "Google Gemini": [
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash"
    ],
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "qwen-2.5-32b"
    ]
}

def highlight_text(text, word_scores):
    """
    Helper to generate HTML with highlighted words based on importance scores.
    Mimics annotated_text without needing the extra library dependency.
    """
    if not word_scores:
        return text
    
    # Sort by length to avoid partial replacement issues (e.g. replacing 'he' inside 'the')
    sorted_words = sorted(word_scores, key=lambda x: len(x[0]), reverse=True)
    
    highlighted = text
    for word, score in sorted_words:
        # Normalize score for opacity (0.2 to 1.0)
        opacity = min(max(score * 2, 0.2), 1.0)
        # Yellow highlight with dynamic opacity
        html_span = f'<span style="background-color: rgba(255, 215, 0, {opacity}); padding: 2px 4px; border-radius: 4px; font-weight: bold; color: black;">{word}</span>'
        
        # Case insensitive replacement using regex to preserve original casing in text
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(html_span, highlighted)
        
    return highlighted

st.set_page_config(page_title="Medical Text AI Agent", layout="wide")

st.title("Medical NLP Agentic System")
st.markdown("### Automated Clinical Text Classification")

with st.sidebar:
    st.header("Configuration")
    
    # MOVED LLM CONFIG HERE so it is accessible for both Training (AutoML) and Inference (XAI)
    st.subheader("LLM Settings (Orchestrator & XAI)")
    llm_provider = st.selectbox("Provider", list(MODEL_OPTIONS.keys()))
    api_key = st.text_input("API Key", type="password")
    available_models = MODEL_OPTIONS[llm_provider]
    model_name = st.selectbox("LLM Model", available_models)
    
    st.markdown("---")
    
    mode = st.radio("Operation Mode", ["Inference (Prediction)", "Training (Orchestration)"])
    
    if not api_key:
        st.warning("No API Key provided.\n\n- Training: Will use Heuristics.\n- Inference: Clinical Explanation disabled.")

# ==========================================
# 1. INFERENCE MODE
# ==========================================
if mode == "Inference (Prediction)":
    st.header("Diagnostic Module (NLP)")
    
    input_method = st.radio(
        "Input Method:",
        ["Manual Text Input", "Select from Test Set"],
        horizontal=True
    )

    target_text = None
    real_label_info = None

    if input_method == "Select from Test Set":
        if os.path.exists("experiments/test_samples.csv"):
            test_df = pd.read_csv("experiments/test_samples.csv")
            
            if test_df.empty:
                st.warning("Test samples file is empty.")
            else:
                selected_idx = st.selectbox(
                    "Select Sample:", 
                    test_df.index,
                    format_func=lambda x: f"Sample {x} (Label: {test_df.iloc[x]['label']}) - {str(test_df.iloc[x]['text'])[:30]}..."
                )
                target_text = str(test_df.iloc[selected_idx]['text'])
                real_label = test_df.iloc[selected_idx]['label']
                real_label_info = f"Class {real_label}"
        else:
            st.warning("test_samples.csv not found. Run training first.")

    elif input_method == "Manual Text Input":
        target_text = st.text_area("Enter Clinical Text / Patient History:", height=150, placeholder="Type patient symptoms, history, or clinical notes here...")

    if target_text:
        st.markdown("### Input Preview")
        st.text_area("Raw Text", target_text, disabled=True, height=100)
        
        if real_label_info:
            st.success(f"Ground Truth: **{real_label_info}**")
       
        col_act = st.columns(1)[0]
        
        if col_act.button("Run Analysis"):
            try:
                if not os.path.exists("experiments/results.json"):
                    st.error("No trained model found. Please run Training first.")
                else:
                    with st.spinner("Analyzing Text & Computing Gradients..."):
                        engine = MedicalTextInference(experiment_dir="experiments")
                        result = engine.predict(target_text)
                    
                    if "error" in result:
                        st.error(result['error'])
                    else:
                        st.success("Analysis Complete")
                        
                        # --- Metrics Display ---
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Predicted Class", f"{result['predicted_class']}")
                        m2.metric("Confidence", result['confidence_score'])
                        m3.metric("Entropy (Uncertainty)", result['entropy'])
                        
                        st.markdown(f"**Uncertainty Status:** {result['uncertainty_status']}")
                        
                        # --- Probability Chart ---
                        st.bar_chart(pd.DataFrame(
                            result['class_probabilities'], 
                            index=[f"Class {i}" for i in range(len(result['class_probabilities']))], 
                            columns=["Probability"]
                        ))

                        # --- Match Verification ---
                        if real_label_info:
                            pred_cls = result['predicted_class']
                            real_cls = int(real_label_info.split()[-1])
                            if pred_cls == real_cls:
                                st.success("✅ Prediction matches Ground Truth.")
                            else:
                                st.error(f"❌ Prediction mismatch. Real: {real_cls}, Predicted: {pred_cls}")
                        
                        st.markdown("---")
                        
                        # ==========================================
                        # 5. UI VISUALIZATION (XAI & EXPLANATION)
                        # ==========================================
                        st.subheader("🧠 Explainable AI (XAI) Analysis")
                        
                        # A. Word Attribution Visualization
                        st.markdown("**1. Gradient-weighted Text Attribution** (Darker Yellow = Higher Influence)")
                        
                        if "word_attributions" in result and result['word_attributions']:
                            # Generate Highlighted HTML
                            html_text = highlight_text(target_text, result['word_attributions'])
                            st.markdown(f'<div style="font-family: sans-serif; line-height: 1.6; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">{html_text}</div>', unsafe_allow_html=True)
                            
                            with st.expander("See Top Influential Keywords"):
                                st.write(result['word_attributions'])
                        else:
                            st.info("No gradient attributions available (Model might not support XAI or calculation failed).")
                        
                        st.markdown("<br>", unsafe_allow_html=True)

                        # B. LLM Generated Clinical Explanation
                        st.markdown("**2. AI Clinical Interpretation**")
                        
                        if "top_influential_words" in result and result['top_influential_words']:
                            if api_key:
                                with st.spinner("Generating clinical explanation based on findings..."):
                                    # Initialize Orchestrator just for the explanation generation
                                    orch = Orchestrator(provider=llm_provider, api_key=api_key, model_name=model_name)
                                   
                                    explanation = orch.generate_explanation(
                                        text_snippet=target_text,
                                        predicted_class=result['predicted_class'],
                                        keywords=result['top_influential_words']
                                    )
                                    
                                    st.info(f"**🩺 Doctor's Note (AI Generated):**\n\n{explanation}")
                            else:
                                st.warning("Please provide an API Key in the sidebar to generate the Clinical Explanation report.")
                        else:
                            st.write("Not enough key features identified to generate a report.")

            except Exception as e:
                st.error(f"Critical Error: {e}")

# ==========================================
# 2. TRAINING MODE
# ==========================================
else:
    st.header("Agentic Training Loop (NLP)")
    
    st.subheader("1. Data Configuration")
    data_path = st.text_input("CSV Path (Must contain 'text' and 'label' columns)", "data/medical_ready.csv")
    
    st.markdown("**Dataset Split Ratios (%)**")
    s1, s2, s3 = st.columns(3)
    with s1:
        train_pct = st.number_input("Train %", 10, 90, 70, 5)
    with s2:
        val_pct = st.number_input("Validation %", 5, 50, 15, 5)
    with s3:
        test_pct = 100 - (train_pct + val_pct)
        st.metric("Test % (Auto-calc)", f"{test_pct}%")

    if test_pct < 0:
        st.error("Sum of Train and Validation exceeds 100%.")
    
    st.markdown("---")
    st.subheader("2. Budget & Constraints")
    
    # [FEATURE] Epoch Limiter UI
    c_trials, c_seed, c_epoch_limit = st.columns(3)
    with c_trials:
        num_trials = st.slider("Max Trials (Budget)", 1, 10, 3)
    with c_seed:
        random_seed = st.number_input("Random Seed", 1, 10000, 42)
    with c_epoch_limit:
        max_user_epochs = st.slider("Max Epoch Limit", 1, 20, 5, help="Hard limit on epochs per trial set by user")
    
    # Start Button
    start_btn = st.button("Start Orchestration")
    
    # [FEATURE] Stop Button State Management
    if "stop_training" not in st.session_state:
        st.session_state.stop_training = False

    def stop_callback():
        st.session_state.stop_training = True

    if start_btn:
        st.session_state.stop_training = False # Reset stop flag on new run
        
        if test_pct < 0:
            st.stop()
        
        st.write(f"Initializing Agentic System (Seed: {random_seed})...")
        os.makedirs("experiments", exist_ok=True)
        
        memory = ExperienceMemory()
        # Orchestrator uses the API key from the global sidebar now
        orchestrator = Orchestrator(max_trials=num_trials, provider=llm_provider, api_key=api_key, model_name=model_name)
        budget = BudgetManager(max_trials=num_trials, max_time_hours=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if not os.path.exists(data_path):
                st.error("Dataset CSV not found.")
            else:
                # 1. Load Data Frame Once to determine Splits
                df_full = pd.read_csv(data_path)
                if 'text' not in df_full.columns or 'label' not in df_full.columns:
                    st.error("CSV must have 'text' and 'label' columns.")
                    st.stop()
                    
                targets = df_full['label'].values
                indices = np.arange(len(df_full))
                
                # Split Indices (Stratified)
                train_idx, temp_idx = train_test_split(indices, train_size=train_pct/100, stratify=targets, random_state=random_seed)
                
                remaining_pct = val_pct + test_pct
                if test_pct == 0:
                    val_idx = temp_idx
                    test_idx = np.array([], dtype=int)
                else:
                    val_relative_ratio = val_pct / remaining_pct
                    temp_targets = targets[temp_idx]
                    val_idx, test_idx = train_test_split(temp_idx, train_size=val_relative_ratio, stratify=temp_targets, random_state=random_seed)
                
                # Save Test Set for Inference
                test_df_save = df_full.iloc[test_idx][['text', 'label']]
                test_df_save.to_csv("experiments/test_samples.csv", index=False)
                st.success(f"Data Split Created. Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
                
                # Class Weights calculation
                from sklearn.utils.class_weight import compute_class_weight
                y_train = targets[train_idx]
                unique_classes = np.unique(y_train)
                class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_train)
                
                # Loop
                progress_bar = st.progress(0)
                status_box = st.empty()

                # [FEATURE] Global Stop Button
                st.button("⛔ STOP TRAINING NOW", on_click=stop_callback, key="global_stop")

                for i in range(num_trials):
                    # Check Stop Request
                    if st.session_state.stop_training:
                        st.warning("Training stopped by user request.")
                        break

                    budget.start_trial()
                    trial_id = budget.current_trial_count
                    status_box.info(f"Trial {trial_id}/{num_trials}: Planning...")
                    
                    # Plan
                    plan, reasoning = orchestrator.plan_next_trial(memory)
                    
                    # [FEATURE] Apply Epoch Limit
                    if 'epochs' in plan:
                        original_epochs = plan['epochs']
                        plan['epochs'] = min(original_epochs, max_user_epochs)
                        if plan['epochs'] < original_epochs:
                            reasoning += f" [Note: Epochs capped from {original_epochs} to {plan['epochs']} by user]"

                    if not plan:
                        st.warning(reasoning)
                        break
                    
                    with st.expander(f"Trial {trial_id}: {plan['model_name']} ({plan['strategy']})", expanded=True):
                        st.write(f"**Reasoning:** {reasoning}")
                        st.json(plan)
                        
                        # [FEATURE] Live Chart Container
                        st.markdown("### 📈 Live Training Progress")
                        chart_placeholder = st.empty()
                        live_metrics_df = pd.DataFrame(columns=["Train Loss", "Val Loss", "Val Acc"])

                        try:
                            # Dynamic Tokenizer & Dataset Creation per Trial
                            model_mapping = {'bert': 'bert-base-uncased', 'biobert': 'dmis-lab/biobert-v1.1', 'roberta': 'roberta-base'}
                            hf_name = model_mapping.get(plan['model_name'], plan['model_name'])
                            
                            tokenizer = AutoTokenizer.from_pretrained(hf_name)
                            
                            # Create Subsets using the fixed indices
                            full_ds_trial = MedicalTextDataset(data_path, tokenizer)
                            train_subset = Subset(full_ds_trial, train_idx)
                            val_subset = Subset(full_ds_trial, val_idx)
                            
                            train_loader = DataLoader(train_subset, batch_size=plan['batch_size'], shuffle=True)
                            val_loader = DataLoader(val_subset, batch_size=plan['batch_size'], shuffle=False)
                            
                            # Model
                            model = get_base_model(plan['model_name'], num_classes=len(unique_classes)).to(device)
                            model = apply_tuning_strategy(model, plan['strategy'])
                            model = model.to(device)
                            
                            # Trainer initialization
                            trainer = ModelTrainer(device=device, class_weights=class_weights.tolist())

                            # --- MANUAL TRAINING LOOP FOR LIVE UPDATES ---
                            lr = plan.get('lr', 2e-5)
                            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
                            
                            best_acc = 0.0
                            best_metrics = {}
                            history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
                            epochs = plan.get('epochs', 3)

                            for epoch in range(epochs):
                                if st.session_state.stop_training:
                                    break
                                
                                # 1. Train
                                train_loss, train_acc = trainer.train_one_epoch(model, train_loader, optimizer)
                                
                                # 2. Evaluate
                                val_metrics = trainer.evaluate(model, val_loader)
                                
                                # 3. Update History
                                history['train_loss'].append(train_loss)
                                history['train_acc'].append(train_acc)
                                history['val_loss'].append(val_metrics['val_loss'])
                                history['val_acc'].append(val_metrics['accuracy'])
                                
                                # 4. Update Live Chart
                                new_row = pd.DataFrame({
                                    "Train Loss": [train_loss],
                                    "Val Loss": [val_metrics['val_loss']],
                                    "Val Acc": [val_metrics['accuracy']]
                                }, index=[f"Ep {epoch+1}"])
                                live_metrics_df = pd.concat([live_metrics_df, new_row])
                                chart_placeholder.line_chart(live_metrics_df)
                                
                                st.caption(f"Epoch {epoch+1}/{epochs} | Val Acc: {val_metrics['accuracy']:.4f}")
                                
                                # 5. Keep Best
                                if val_metrics['accuracy'] > best_acc:
                                    best_acc = val_metrics['accuracy']
                                    best_metrics = val_metrics.copy()
                            
                            # --- END MANUAL LOOP ---

                            if not best_metrics: best_metrics = val_metrics
                            best_metrics['history'] = history
                            
                            st.success(f"Trial Completed. Best Accuracy: {best_metrics['accuracy']:.4f}")
                            
                            # Evaluation Plots (Static Final Reports)
                            c1, c2 = st.columns(2)
                            
                            hist_path = f"experiments/history_trial_{trial_id}.png"
                            Evaluator.plot_training_history(history, hist_path) 
                            c1.image(hist_path, caption="Full Learning Curve")
                            
                            cm_path = f"experiments/cm_trial_{trial_id}.png"
                            Evaluator.plot_confusion_matrix(np.array(best_metrics['conf_matrix']), [f"C{k}" for k in unique_classes], cm_path)
                            c2.image(cm_path, caption="Confusion Matrix")

                            plan['num_classes'] = len(unique_classes)
                            memory.add_experience(trial_id, plan, best_metrics)
                            torch.save(model.state_dict(), f"experiments/model_trial_{trial_id}.pth")
                            
                            del model, tokenizer
                            torch.cuda.empty_cache()

                        except Exception as e:
                            st.error(f"Trial Failed: {str(e)}")
                            memory.add_experience(trial_id, plan, {'accuracy': 0, 'error': str(e)})
                    
                    progress_bar.progress((i + 1) / num_trials)
                
                memory.save_memory("experiments/results.json")
                
                if st.session_state.stop_training:
                    status_box.warning("Orchestration Stopped by User.")
                else:
                    status_box.success("Orchestration Completed!")
        
        except Exception as e:
            st.error(f"System Error: {str(e)}")