import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np

class UBFCSeperate78Dataset(Dataset):
    def __init__(self, data, n_frames=10, frame_per_file=6320, transform=None):
        self.input_image = glob.glob(os.path.join(data, "images/*.npy")) 
        self.input_image.sort()
        self.gt = glob.glob(os.path.join(data, "gt/*.npy"))
        self.gt.sort()
        self.n_frames = n_frames
        # count the number of frames in the first file. Its same across the files.
        self.frame_per_file = frame_per_file
        self.batch_per_file = self.frame_per_file // self.n_frames

    
    def __len__(self):
        return len(self.input_image) * self.batch_per_file
    
    def __getitem__(self, idx):
        image_id = idx // self.batch_per_file
        index = idx % self.batch_per_file

        # load image numpy file
        input_numpy = np.load(self.input_image[image_id])
        output_numpy = np.load(self.gt[image_id])
        # convert output to tensor and add dim
        output_numpy = torch.from_numpy(output_numpy).unsqueeze(dim=1)

        batch = input_numpy[self.n_frames * index: self.n_frames * index + self.n_frames]
        # first 3 channels for motion, last 3 channels for appearance
        motion_data, appearance_data = batch[:,:3,:,:], batch[:,-3:,:,:]
        target = output_numpy[self.n_frames * index: self.n_frames * index + self.n_frames]

        return {"motion_data": motion_data, "appearance_data": appearance_data, "target": target}
