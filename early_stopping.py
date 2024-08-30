class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, loss):
        if self.best_loss is not None and loss > self.best_loss + self.min_delta:
            self.counter += 1
        else:
            self.counter = 0
            if self.best_loss is None or loss < self.best_loss:
                self.best_loss = loss

    def early_stop(self):
        return self.counter >= self.patience