import torch


def concatenate_batch(batch):
    motion_data = torch.cat([torch.from_numpy(item["motion_data"]) for item in batch], dim=0)
    appearance_data = torch.cat([torch.from_numpy(item["appearance_data"]) for item in batch], dim=0)
    target = torch.cat([item["target"].float() for item in batch], dim=0)
    return {"motion_data": motion_data, "appearance_data": appearance_data, "target": target}


def load_data_loader(data_set, dataloader_args):
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn = concatenate_batch, **dataloader_args)
    return data_loader
