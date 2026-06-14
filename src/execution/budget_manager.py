import time

class BudgetManager:
    def __init__(self, max_trials=10, max_time_hours=2):
        self.max_trials = max_trials
        self.max_time_sec = max_time_hours * 3600
        self.start_time = time.time()
        self.current_trial_count = 0

    def start_trial(self):
        self.current_trial_count += 1
        print(f"[Budget] Starting trial {self.current_trial_count}/{self.max_trials}...")

    def is_budget_exhausted(self):
        if self.current_trial_count >= self.max_trials:
            print("[Budget] Max trials reached.")
            return True
        
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_time_sec:
            print(f"[Budget] Time limit reached ({elapsed/3600:.2f} hours).")
            return True
            
        return False

    def get_remaining_budget(self):
        return {
            'trials_left': self.max_trials - self.current_trial_count,
            'time_left_sec': self.max_time_sec - (time.time() - self.start_time)
        }