import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .evaluator import Evaluator
import torch.nn.functional as F

class ModelTrainer:
    def __init__(self, device, class_weights=None):
        self.device = device
        
        if class_weights is not None:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
            print(f"[Trainer] Using Weighted Cross Entropy: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_one_epoch(self, model, loader, optimizer):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc="Training", leave=False)
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['grade_label'].to(self.device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
        return running_loss / len(loader), correct / total

    def evaluate(self, model, loader, return_raw=False):
        model.eval()
        all_outputs = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['grade_label'].to(self.device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                all_outputs.append(logits)
                all_targets.append(labels)
        
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        
        # Passing targets as group_targets placeholder since strictly medical groups aren't defined here
        metrics = Evaluator.compute(all_outputs, all_targets, all_targets)
        metrics['val_loss'] = total_loss / len(loader)
        
        if return_raw:
            return metrics, F.softmax(all_outputs, dim=1), all_targets
            
        return metrics

    def run_training(self, model, train_loader, val_loader, config):
        print(f"[Trainer] Starting training on {self.device}...")
        lr = config.get('lr', 2e-5) # Lower learning rate is better for BERT
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        
        best_acc = 0.0
        best_metrics = {}
        epochs = config.get('epochs', 3)
        
        history = {
            'train_loss': [], 
            'train_acc': [], 
            'val_loss': [], 
            'val_acc': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(model, train_loader, optimizer)
            val_metrics = self.evaluate(model, val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                best_metrics = val_metrics.copy()
        
        if not best_metrics:
            best_metrics = val_metrics
            
        best_metrics['history'] = history
                
        return best_metrics