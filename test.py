from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import torch
from datasets.grain_space import GrainDataset
# init = GrainDataset(root='/home/ruiqi/GrainSpace/data/WHEAT_R1-14_G600/train')

# print(init.class_to_idx)

transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.4254, 0.3929, 0.2554],
                                  [0.1206, 0.1556, 0.1535])
            ])
            
train_path = 'GrainSpace/WHEAT_R1-14_G600/train'
train_dataset = GrainDataset(train_path, transform=transform)
# print(train_dataset.class_to_idx['NOR'])
# print(train_dataset.targets)

count = {}
for target in train_dataset.targets:
    if target in count:
        count[target] += 1
    else:
        count[target] = 1
    
    
print(count)

# print(train_dataset.classes)
# print(train_dataset.imgs)

index = 0
N_CLASSES_PER_TASK = 2
N_TASKS = 4

print(len(train_dataset))

for t in range(N_TASKS):
    print(f"task {t}")
    train_mask = torch.logical_and(torch.tensor(train_dataset.targets) >= index, torch.tensor(train_dataset.targets) < index + N_CLASSES_PER_TASK)
    # train_mask = np.logical_and(np.array(train_dataset.targets) >= index,
    #                             np.array(train_dataset.targets) < index + N_CLASSES_PER_TASK)
    print(train_mask)
    
    # find first true and last true
    first = -1
    last = -1
    for i, mask in enumerate(train_mask):
        if first == -1 and mask:
            first = i
        if mask:
            last = i
    print(f"first:{first}")
    print(f"last:{last}")
            
    # extract the data and targets for the current task
    # print(train_mask.nonzero())
    train_indices = train_mask.nonzero().reshape(-1)
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_subset,
                              batch_size=32, shuffle=True, num_workers=4)
    # train_subset = train_dataset                   
    # train_subset.imgs = np.array(train_dataset.imgs)[train_mask]
    # train_subset.targets = np.array(train_dataset.targets)[train_mask]
    # print(len(imgs))
    # print(len(targets))
    # print(len(train_subset))
    # print(train_subset.transform)
    
    # train_loader = DataLoader(train_subset,
    #                           batch_size=32, shuffle=True, num_workers=4)
    count = {}
    for batch_idx, sample in enumerate(train_loader):
        (data, target, not_arg) = sample
        print(data.shape)
        print(target.shape)
        print(target)
        break
        for tmp in target:
            tmp = tmp.item()
            if tmp in count:
                count[tmp] += 1
            else:
                count[tmp] = 1
    print(count)
    index += N_CLASSES_PER_TASK

