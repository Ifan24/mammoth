from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision.models as models
from argparse import Namespace
import time
from torchmetrics.classification import F1Score


class GrainDataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(GrainDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.not_aug_transform = transforms.Compose([
                transforms.Resize((256, 144)),
                transforms.ToTensor(),
            ])
            
        # don't lazy load
        # self.subset = MySubset(self)
        # self.data = []
        # start = time.time()
        # for img in self.imgs:
        #     self.data.append(self.loader(img[0]))
        # end = time.time()
        # 33s
        # print(end - start)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        original_img = self.loader(path)
        not_aug_img = self.not_aug_transform(original_img)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target, not_aug_img
        
        
def store_grain_masked_loaders(train_dataset: Dataset, test_dataset: Dataset,
                         setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    
    # create a mask for the current task
    train_mask = torch.logical_and(torch.tensor(train_dataset.targets) >= setting.i, torch.tensor(train_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = torch.logical_and(torch.tensor(test_dataset.targets) >= setting.i, torch.tensor(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK)
    
    # extract the data and targets for the current task
    train_indices = train_mask.nonzero().reshape(-1)
    train_subset = Subset(train_dataset, train_indices)
    train_loader = DataLoader(train_subset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
                              
    test_indices = test_mask.nonzero().reshape(-1)
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
        
        
    # add all previous test loaders (CL scenario)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader
class SequentialGrainSpace(ContinualDataset):

    NAME = 'seq-GrainSpace'
    SETTING = 'class-il'
    N_CLASSES = 7
    N_CLASSES_PER_TASK = 2
    N_TASKS = 4
    # Sequential
    # Task1: AP, BN
    # Task2: BP, FS
    # Task3: MY, NOR
    # Task4: SD
    
    TRANSFORM = transforms.Compose(
            [
             transforms.Resize((256, 144)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.4829, 0.5062, 0.3941],
                                  [0.1385, 0.2133, 0.2385])
            ])
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.init = False
        self.norIndex = 0
        # self.f1 = F1Score(task="multiclass", num_classes=self.N_CLASSES, average="macro")
        
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
                transforms.Resize((256, 144)),
                transforms.ToTensor(),
                self.get_normalization_transform()
        ])
        
        train_path = self.args.dataset_path + '/train'
        val_path = self.args.dataset_path + '/val'
        test_path = self.args.dataset_path + '/test'
        
        train_dataset = GrainDataset(train_path, transform=transform)
        self.norIndex = train_dataset.class_to_idx['NOR']
        self.class_to_idx = train_dataset.class_to_idx
        self.classes = train_dataset.classes
        
        if self.args.validation:
            test_dataset = ImageFolder(val_path, transform=test_transform)
        else:
            test_dataset = ImageFolder(test_path, transform=test_transform)
            
        train, test = store_grain_masked_loaders(train_dataset, test_dataset, self)
        return train, test
        
    def get_current_task_classes(self, k) -> List[int]:
        return list(range(k, k + self.N_CLASSES_PER_TASK))
        
    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialGrainSpace.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        # return resnet18(SequentialGrainSpace.N_CLASSES)
        resnet_model = models.resnet50(pretrained=False)
        linear_size = list(resnet_model.children())[-1].in_features
        resnet_model.fc = nn.Linear(linear_size, SequentialGrainSpace.N_CLASSES)
        return resnet_model
        
    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize([0.4829, 0.5062, 0.3941],
                                         [0.1385, 0.2133, 0.2385])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize([0.4829, 0.5062, 0.3941],
                                [0.1385, 0.2133, 0.2385])
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialGrainSpace.get_batch_size()

class SplitGrainSpaceA(SequentialGrainSpace):

    NAME = 'split-GrainSpaceA'
    SETTING = 'class-il'
    N_CLASSES = 7
    
    N_CLASSES_PER_TASK = 2
    N_TASKS = 6
    # split NORMAL into 6 subsets
    # Task1: NOR1, AP
    # Task2: NOR2, BN
    # Task3: NOR3, BP
    # Task4: NOR4, FS
    # Task5: NOR5, MY
    # Task6: NOR6, SD

        
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
                transforms.Resize((256, 144)),
                transforms.ToTensor(),
                self.get_normalization_transform()
        ])
        
        train_path = self.args.dataset_path + '/train'
        val_path = self.args.dataset_path + '/val'
        test_path = self.args.dataset_path + '/test'
        
        train_dataset = GrainDataset(train_path, transform=transform)
        self.class_to_idx = train_dataset.class_to_idx
        self.classes = train_dataset.classes
        self.norIndex = train_dataset.class_to_idx['NOR']
        
        if self.args.validation:
            test_dataset = ImageFolder(val_path, transform=test_transform)
        else:
            test_dataset = ImageFolder(test_path, transform=test_transform)
            
            
        if not self.init:
            NOR_mask = torch.tensor(train_dataset.targets) == train_dataset.class_to_idx['NOR']
            NOR_indices = NOR_mask.nonzero().reshape(-1)
            # shuffle the index then split it into N_TASKS subsets
            self.train_NOR_subsets = NOR_indices[torch.randperm(NOR_indices.shape[0])].chunk(self.N_TASKS)
            
            # Test set
            NOR_mask = torch.tensor(test_dataset.targets) == test_dataset.class_to_idx['NOR']
            NOR_indices = NOR_mask.nonzero().reshape(-1)
            # shuffle the index then split it into N_TASKS subsets
            self.test_NOR_subsets = NOR_indices.chunk(self.N_TASKS)
            
            self.init = True
           
           
        def gen_loader(dataset, current_NOR_indices, train):
            # find current task index
            task_class = self.i if self.i < dataset.class_to_idx['NOR'] else self.i + 1
            task_mask = torch.tensor(dataset.targets) == task_class
            task_indices = task_mask.nonzero().reshape(-1)
            # concat the indices
            task_indices = torch.cat((current_NOR_indices, task_indices))
            task_subset = Subset(dataset, task_indices)
            task_loader = DataLoader(task_subset,
                                      batch_size=32, shuffle=train, num_workers=4)
            
            return task_loader
        
        train_loader = gen_loader(train_dataset, self.train_NOR_subsets[self.i], True)
        test_loader = gen_loader(test_dataset, self.test_NOR_subsets[self.i], False)
                              
        # add all previous test loaders (CL scenario)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
    
        self.i += 1
    
        return train_loader, test_loader
        
        
    def get_current_task_classes(self, k) -> List[int]:
        classes = []
        classes.append(k if k < self.norIndex else k + 1)
        classes.append(self.class_to_idx['NOR'])
        return classes

class SplitGrainSpaceB(SequentialGrainSpace):

    NAME = 'split-GrainSpaceB'
    SETTING = 'class-il'
    N_CLASSES = 7
    
    N_CLASSES_PER_TASK = 3
    N_TASKS = 3
    # split NORMAL into 3 subsets
    # Task1: NOR1, AP, BN
    # Task2: NOR2, BP, FS
    # Task3: NOR3, MY, SD

        
    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
                transforms.Resize((256, 144)),
                transforms.ToTensor(),
                self.get_normalization_transform()
        ])
        
        train_path = self.args.dataset_path + '/train'
        val_path = self.args.dataset_path + '/val'
        test_path = self.args.dataset_path + '/test'
        
        train_dataset = GrainDataset(train_path, transform=transform)
        self.class_to_idx = train_dataset.class_to_idx
        self.classes = train_dataset.classes
        
        if self.args.validation:
            test_dataset = ImageFolder(val_path, transform=test_transform)
        else:
            test_dataset = ImageFolder(test_path, transform=test_transform)
            
            
        if not self.init:
            NOR_mask = torch.tensor(train_dataset.targets) == train_dataset.class_to_idx['NOR']
            NOR_indices = NOR_mask.nonzero().reshape(-1)
            # shuffle the index then split it into N_TASKS subsets
            self.train_NOR_subsets = NOR_indices[torch.randperm(NOR_indices.shape[0])].chunk(self.N_TASKS)
            
            # Test set
            NOR_mask = torch.tensor(test_dataset.targets) == test_dataset.class_to_idx['NOR']
            NOR_indices = NOR_mask.nonzero().reshape(-1)
            # shuffle the index then split it into N_TASKS subsets
            self.test_NOR_subsets = NOR_indices.chunk(self.N_TASKS)
            
            self.init = True
           
           
        def gen_loader(dataset, current_NOR_indices, train):
            # find current task index
            task_class = self.get_current_task_classes(self.i)
            task_mask = torch.logical_or(torch.tensor(dataset.targets) == task_class[0], torch.tensor(dataset.targets) == task_class[1])
            task_indices = task_mask.nonzero().reshape(-1)
            
            # concat the indices
            task_indices = torch.cat((current_NOR_indices, task_indices))
            task_subset = Subset(dataset, task_indices)
            task_loader = DataLoader(task_subset,
                                      batch_size=32, shuffle=train, num_workers=4)
            
            return task_loader
        
        train_loader = gen_loader(train_dataset, self.train_NOR_subsets[self.i], True)
        test_loader = gen_loader(test_dataset, self.test_NOR_subsets[self.i], False)
                              
        # add all previous test loaders (CL scenario)
        self.test_loaders.append(test_loader)
        self.train_loader = train_loader
    
        self.i += 1
    
        return train_loader, test_loader
        
        
    def get_current_task_classes(self, k) -> List[int]:
        task_classes = [c for c in self.classes if c != 'NOR']
        task_classes = [v for (key, v) in self.class_to_idx.items() if key in task_classes]
        task = np.array_split(task_classes, self.N_TASKS)
        task = [l.tolist() + [self.class_to_idx['NOR']] for l in task]
        return task[k]
        
    