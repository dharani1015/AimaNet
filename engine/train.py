import torch

def train(model, device, train_loader, optimizer, criterion, kbar=None):
    model.train()

    # metrics stats collection
    epoch_loss = 0

    for batch_id, batch in enumerate(train_loader):
        x_motion, x_appearance = batch["motion_data"].to(device), batch["appearance_data"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        pred = model(x_motion, x_appearance)
        loss = criterion(pred, target)

        # calculate gradients
        loss.backward()
        # optimizer
        optimizer.step()

        # metrics
        epoch_loss += loss.item()

        if kbar is not None:
            kbar.update(batch_id, values=[("loss", loss.item())])
    
    epoch_loss /= len(train_loader)
    print(f"Training-----\tEpoch Loss {round(epoch_loss, 4)}")
    return epoch_loss
