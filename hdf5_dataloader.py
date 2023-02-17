import torch
from torch.utils.data import DataLoader
import h5py

class My_H5Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, flip=False):
        super(My_H5Dataset, self).__init__()
        h5_file = h5py.File(file_path , 'r')
        if flip:
            self.feats = h5_file['feats_hflip']
        else:
            self.feats = h5_file['feats']
    

    def __getitem__(self, index): 
        return (torch.from_numpy(self.feats[index,:]).float(),
            )

    def __len__(self):
        return self.feats.shape[0]

train_dataset = My_H5Dataset('/mnt/petrelfs/yangmengping/ICGAN/coco/COCO128_feats_selfsupervised_resnet50.hdf5')
train_loader = DataLoader(dataset = train_dataset, num_workers=0, batch_size=8, shuffle=True)
for data in train_loader:
    noise = data
    print(noise)