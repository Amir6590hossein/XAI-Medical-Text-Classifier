import torch
from transformers import AutoModelForSequenceClassification, AutoConfig

def get_base_model(model_name, num_classes=5, pretrained=True):
    
    model_mapping = {
        'bert': 'bert-base-uncased',
        'biobert': 'dmis-lab/biobert-v1.1',
        'roberta': 'roberta-base'
    }
    
    hf_model_name = model_mapping.get(model_name, model_name)
    
    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,
            num_labels=num_classes
        )
    else:
        config = AutoConfig.from_pretrained(hf_model_name, num_labels=num_classes)
        model = AutoModelForSequenceClassification.from_config(config)
        
    return model