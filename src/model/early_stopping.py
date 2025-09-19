class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.min_val = float('inf')
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.min_val:
            self.min_val = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0
        elif val_loss > (self.min_val + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)