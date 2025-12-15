import torch

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device
    ):
    
    model.train()
    
    running_loss = 0
    correct = 0
    total = 0
    
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * xb.size(0)
        predicted_labels = preds.argmax(dim=1)
        correct += (predicted_labels == yb).sum().item()
        total += yb.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def eval_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
    ):
    
    model.eval()
    
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            preds = model(xb)
            loss = criterion(preds, yb)
            
            running_loss += loss.item() * xb.size(0)
            predicted_labels = preds.argmax(dim=1)
            correct += (predicted_labels == yb).sum().item()
            total += yb.size(0)
            
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc  = correct / total if total > 0 else 0.0
    
    return epoch_loss, epoch_acc