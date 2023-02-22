import torch
from torch.utils.data import Dataset

class UBFCDataset(Dataset):
    def __init__(self, input_numpy, output_numpy, transform=None):
        self.input_numpy = input_numpy
        self.output_numpy = torch.from_numpy(output_numpy).unsqueeze(dim=1)
        self.transform = transform
    
    def __len__(self):
        return len(self.input_numpy) // 10
    
    def __getitem__(self, index):
        batch = self.input_numpy[10*index: 10*index + 10]
        motion_data, appearance_data = batch[:,:3,:,:], batch[:,-3:,:,:]
        target = self.output_numpy[10*index: 10*index + 10]

        return {"motion_data": motion_data, "appearance_data": appearance_data, "target": target}
