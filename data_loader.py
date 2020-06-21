from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def data_loader(size=1, target_idx=0, batch_size=64, shuffle=True, num_workers=0):
    path = '../db/'
    if target_idx == 0:
        path += 'data_source/'
    elif target_idx == 1:
        path += 'data_target1/'
    elif target_idx == 2:
        path += 'data_target2/'
    elif target_idx == 3:
        path += 'data_target3/'
    else:
        path += 'data_source/'

    if size == 1:
        path += 'train/'
    else:
        path += 'val/'

    composed = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=path, transform=composed)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl

if __name__ == "__main__":
    train_loader = data_loader(size=1, target_idx=0, shuffle=True, batch_size=64)
    val_loader = data_loader(size=0, target_idx=0, shuffle=True, batch_size=64)
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples
        print(data.shape, target.shape)

        break

    for batch_idx, samples in enumerate(val_loader):
        data, target = samples
        print(data.shape, target.shape)

        break