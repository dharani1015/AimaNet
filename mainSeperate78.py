import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.UBFCSeperate78 import UBFCSeperate78Dataset
from data.ubfc_data_loader import load_data_loader
from utils.load_numpy_split import load_numpy_and_split
from models.model import TemporalShiftCAN
from engine.train import train
from engine.test import test
from utils.pkbar import Kbar



def model_train(model, device, train_loader, test_loader, optimizer, criterion, model_checkpoint, scheduler=None, EPOCHS=8):
    train_loss, test_loss = [], []
    best_loss = np.inf

    for epoch in range(EPOCHS):
        print(f"Epoch run- {epoch+1}")

        kbar = Kbar(target=len(train_loader), epoch=epoch, num_epochs=EPOCHS, width=8, always_stateful=False)

        train_l = train(model, device, train_loader, optimizer, criterion, kbar)
        test_l = test(model, device, test_loader, criterion, kbar)
        train_loss.append(train_l)
        test_loss.append(test_l)

        if test_l < best_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'loss': test_l
            }, model_checkpoint)
            best_loss = test_l

            #cv edits start
            print("Best Test Loss Found: Saving the train and test losses...")
            np.save(os.path.join(results_dir, "train_loss.npy"), train_loss)
            np.save(os.path.join(results_dir, "test_loss.npy"), test_loss)
            #cv edits end
    
    print("Saving Final train and test losses...")
    np.save(os.path.join(results_dir, "train_loss.npy"), train_loss)
    np.save(os.path.join(results_dir, "test_loss.npy"), test_loss)

    return train_loss, test_loss 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="./", help='file path of data')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=3, help='size of each batch to run')
    opt = parser.parse_args()

    # enables the inbuilt CUDNN auto-tuner to speed up training process
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    ##### Setting parameters #####
    data_path = opt.data_path
    EPOCHS = opt.epochs
    batch_size = opt.batch_size

    # setting up data, result locations
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    if not os.path.exists("results"):
        os.mkdir("results")
    results_dir = 'results'
    # results path
    filename = (f'best_model_checkpoint_{time.strftime("%Y-%m-%d")}.pt')
    model_checkpoint = os.path.join(results_dir, filename)
    


    dataloader_args = dict(shuffle=False, batch_size=batch_size, num_workers=1, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Available device: {device}\n")
    #####

    # Create Dataset
    ubfc_train_dataset = UBFCSeperate78Dataset(train_path, n_frames=10, frame_per_file=6320)
    ubfc_test_dataset = UBFCSeperate78Dataset(test_path, n_frames=10, frame_per_file=6320)
    

    # load dataloader
    ubfc_train_loader = load_data_loader(ubfc_train_dataset, dataloader_args)
    ubfc_test_loader = load_data_loader(ubfc_test_dataset, dataloader_args)

    # model creation
    ts_can_model = TemporalShiftCAN(n_frame=10, in_channels=3, out_channels_1=32, out_channels_2=64, kernel_size=(3,3), hidden_size=128)
    ts_can_model = ts_can_model.to(device)

    # optimization algorithm
    optimizer = optim.Adadelta(ts_can_model.parameters())
    # loss function
    criterion = nn.MSELoss()

    print(f"Starting training for {EPOCHS} epochs\n")

    train_loss, test_loss = model_train(ts_can_model, device, 
                            ubfc_train_loader, ubfc_test_loader, optimizer, criterion, 
                            model_checkpoint, scheduler=None, EPOCHS=EPOCHS)
