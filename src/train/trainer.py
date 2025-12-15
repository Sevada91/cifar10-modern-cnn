import copy
import torch
from src.train.engine import train_one_epoch, eval_one_epoch

class EarlyStopper():
    def __init__(self, patience: int=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.best_epoch = -1
        self.best_state = None
        self.count = 0
    
    def step(self, model, val_loss, epoch):
        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.count = 0
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            return False
        else:
            self.count += 1
            return self.count >= self.patience
        
def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs :int=20, 
    early_stopper: EarlyStopper | None=None, 
    scheduler=None,
    verbose: bool=True
):
    """
    Full training loop with optional early stopping.
    Returns:
        history dict with loss/acc curves.
    """
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve,  val_acc_curve  = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        val_loss, val_acc = eval_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device
        )
        
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                try:  
                    scheduler.step(val_loss)
                except TypeError:
                    scheduler.step()
        
        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)
        
        if verbose:
            print(
                f"EPOCH {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} Acc: {val_acc:.6f} "
            )
        
        if early_stopper is not None:
            if early_stopper.step(model, val_loss, epoch):
                if verbose:
                    print(
                        f"Early stop at epoch {epoch+1} | "
                        f"Best epoch {early_stopper.best_epoch+1}"
                    )
                break
        
    if early_stopper is not None and early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    history = {
        'train_loss': train_loss_curve,
        'val_loss': val_loss_curve,
        'train_acc': train_acc_curve,
        'val_acc': val_acc_curve
    }
        
    return history