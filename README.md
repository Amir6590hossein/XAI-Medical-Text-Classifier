# Medical NLP Agentic System

## About The Project

This project is developed as the **Natural Language Processing (NLP) course project** at Iran University of Science and Technology (IUST). An **Agentic System** is designed and implemented that automatically manages the training process of language models for medical text classification.

### Project Objectives

- Automating hyperparameter optimization using LLM Orchestrator
- Reducing computational costs using PEFT methods (LoRA and Adapter)
- Increasing transparency and reliability by implementing XAI (Integrated Gradients)
- Providing an interactive user interface for physicians and researchers

### Key Features

| Feature | Description |
|---------|-------------|
| Intelligent Orchestrator | Using LLM (Gemini/Groq/Perplexity) to suggest optimal strategies |
| Experience Memory | Storing and analyzing experiment history for learning from past experiences |
| PEFT | Implementing LoRA and Adapter to reduce 99% of trainable parameters |
| XAI with Captum | Extracting influential keywords using Integrated Gradients |
| Intelligent Clinical Report | Generating medical explanations by LLM for model predictions |
| Interactive Dashboard | Training and inference with Streamlit and real-time charts |

### System Architecture

```
+-----------------------------------------------------------------------------------+
|                              Streamlit UI                                         |
+-----------------------------------------------------------------------------------+
|                              Agentic Core                                         |
|  +-------------+  +-------------+  +---------------------+                       |
|  | Orchestrator|  |   Memory    |  |   Budget Manager    |                       |
|  |   (LLM)     |  | (JSON Store)|  | (Trials/Time Limit) |                       |
|  +------+------+  +------+------+  +----------+----------+                       |
+---------+-------------+--------------------+--------------------------------------+
          |             |                    |
          v             v                    v
+-----------------------------------------------------------------------------------+
|                             Model Layer                                           |
|  +------+ +--------+ +---------+ +------+ +---------+                           |
|  | BERT | | BioBERT| | RoBERTa | | LoRA | | Adapter |                           |
|  +------+ +--------+ +---------+ +------+ +---------+                           |
+-----------------------------------------------------------------------------------+
|                            Execution Layer                                       |
|  +----------+ +--------+ +----------+ +----------+                             |
|  | Trainer  | |Evaluator| | XAI (IG) | |Inference |                             |
|  +----------+ +--------+ +----------+ +----------+                             |
+-----------------------------------------------------------------------------------+
```

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Minimum 8GB RAM (16GB recommended)
- (Optional) GPU with CUDA for faster training

### Installation Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/XAI-Medical-Text-Classifier.git
cd XAI-Medical-Text-Classifier

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

### Dataset Structure

The CSV file must contain two columns:
- `text`: Medical text (report, history, etc.)
- `label`: Numeric label for classification

```csv
text,label
"Patient presents with chest pain and shortness of breath...",0
"MRI shows signs of multiple sclerosis...",1
```

## Project Structure

```
Medical-NLP-Agentic-System/
│
├── app.py                      # Main Streamlit application
├── inference.py                # Inference and XAI module
│
├── src/
│   ├── agent_core/
│   │   ├── orchestrator.py     # LLM Orchestrator
│   │   └── experience_memory.py # Experience memory
│   │
│   ├── model_layer/
│   │   ├── model_factory.py    # Base model factory
│   │   └── tuners.py           # LoRA and Adapter implementations
│   │
│   ├── data_layer/
│   │   └── dataset_loader.py   # Data loading and tokenization
│   │
│   ├── execution/
│   │   ├── trainer.py          # Training loop
│   │   ├── evaluator.py        # Evaluation metrics
│   │   └── budget_manager.py   # Budget management
│   │
│   └── utils/
│       └── reproducibility.py  # Reproducibility settings
│
├── experiments/                # (Auto-created) Experiment results
│   ├── results.json           # Experiment history
│   ├── test_samples.csv       # Test samples
│   └── model_trial_*.pth      # Model weights
│
├── docs/                       # Project documentation
│   ├── backend_Document.pdf   # Backend technical documentation
│   └── UI_Document.pdf        # UI documentation
│
└── requirements.txt            # Dependencies
```

## User Guide

### Mode 1: Training Mode

1. In the sidebar, select **Training (Orchestration)**
2. Enter the path to your CSV file containing the data
3. Set the data split ratios (Train/Val/Test)
4. Configure the budget (Max Trials) and epoch limit
5. (Optional) Enter API Key for LLM Orchestrator
6. Click **Start Orchestration**

The system will automatically run multiple trials and save the best model.

### Mode 2: Inference Mode

1. In the sidebar, select **Inference (Prediction)**
2. Enter medical text or select from test samples
3. Click **Run Analysis**
4. Output includes:
   - Predicted class and confidence score
   - Entropy and uncertainty status
   - Highlighted keywords (XAI)
   - Clinical report generated by LLM

## Experimental Results

| Trial | Model | Strategy | Learning Rate | Epochs | Accuracy |
|-------|-------|----------|---------------|--------|----------|
| 1 | BioBERT | LoRA | 2e-5 | 3 | 48.26% |
| 2 | BioBERT | LoRA | 2e-4 | 5 | 53.39% |
| 3 | BioBERT | LoRA | 2e-4 | 5 | 49.91% |

> Note: Best accuracy of 53.39% achieved in Trial 2. Class imbalance was identified as the main challenge.

## Technologies Used

| Technology | Purpose |
|------------|---------|
| PyTorch | Main deep learning framework |
| Transformers | BERT, BioBERT, RoBERTa models |
| Captum | Integrated Gradients implementation for XAI |
| Streamlit | Interactive user interface |
| Google Gemini / Groq / Perplexity | LLM Orchestrator and clinical report generation |
| scikit-learn | Evaluation metrics and label encoding |

## Documentation

Two complete documents are available in the `docs/` folder:

1. **backend_Document.pdf** - Technical documentation including:
   - System architecture and module interactions
   - LoRA and Adapter implementations
   - Evaluation and XAI algorithms
   - Class and function implementation details

2. **UI_Document.pdf** - User interface documentation including:
   - Step-by-step software usage guide
   - Screenshots of training and inference environments
   - Analysis of successful and failed cases (with XAI)
   - Model output interpretation

## Contributing

This is an academic project, but contributions are welcome:
- Bug reports (Issues)
- Improvement suggestions
- Pull requests

## Author

**AmirHossein Amjadian**
- Computer Engineering Student - Iran University of Science and Technology
- Course: Natural Language Processing (NLP)
- First Semester 2025-2026

**Supervisors:**
- Dr. Etemadi
- Dr. Minaei
