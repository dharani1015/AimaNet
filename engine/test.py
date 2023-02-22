import torch
def test(model, device, test_loader, criterion, kbar=None):
    model.eval()

    # metrics stats collection
    epoch_loss = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(test_loader):
            x_motion, x_appearance = batch["motion_data"].to(device), batch["appearance_data"].to(device)
            target = batch["target"].to(device)
            
            pred = model(x_motion, x_appearance)
            loss = criterion(pred, target)

            # metrics
            epoch_loss += loss.item()
    
    epoch_loss /= len(test_loader)

    if kbar is not None:
        kbar.add(1, values=[("test_loss", epoch_loss)]) 

    print(f"Inference-----\tEpoch Loss {round(epoch_loss, 4)}")
    return epoch_loss
