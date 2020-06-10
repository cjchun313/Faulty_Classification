from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def data_loader(train=True, batch_size=64, shuffle=True, num_workers=0):
    composed = transforms.Compose([transforms.ToTensor()])
    if train is True:
        dataset = datasets.ImageFolder(root='../db/data_source/train/', transform=composed)
    else:
        dataset = datasets.ImageFolder(root='../db/data_source/val/', transform=composed)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl

def target_data_loader(target_idx=0, batch_size=64, shuffle=True, num_workers=0):
    composed = transforms.Compose([transforms.ToTensor()])
    if target_idx == 0:
        dataset = datasets.ImageFolder(root='../db/data_target1/', transform=composed)
    elif target_idx == 1:
        dataset = datasets.ImageFolder(root='../db/data_target2/', transform=composed)
    elif target_idx == 2:
        dataset = datasets.ImageFolder(root='../db/data_target3/', transform=composed)
    else:
        dataset = datasets.ImageFolder(root='../db/data_target1/', transform=composed)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl

if __name__ == "__main__":
    composed = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root='../db/data_target1/train/', transform=composed)
    val_dataset = datasets.ImageFolder(root='../db/data_source/val/', transform=composed)
    print(train_dataset, val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        print(data.shape, target.shape)

        break