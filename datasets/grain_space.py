from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10

from datasets.seq_tinyimagenet import base_path
from datasets.transforms.denormalization import DeNormalize
from datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from datasets.utils.validation import get_train_val

import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision.models as models


class GrainDataset(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(GrainDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.not_aug_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ])
        
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
        
        
# class GrainDataset(Dataset):
#     def __init__(self, root, transform=None, target_transform=None):
#         self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
#         self.transform = transform
#         self.target_transform = target_transform
#         self.data_dir = root
#         self.classes = os.listdir(root)
#         self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
#         self.samples = [(c, os.path.join(c, filename)) for c in self.classes
#                         for filename in os.listdir(os.path.join(root, c))]
        
#         # all labels
#         self.targets = [self.class_to_idx[i[0]] for i in self.samples]
#         self.data = []
        
#         for sample in self.samples:
#             image = Image.open(os.path.join(self.data_dir, sample[1]))
#             image = self.transform(image)
#             self.data.append(image)
        
#         # turn into numpy array
#         # self.data = np.vstack(self.data).reshape(-1, 3, 224, 224)
#         # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
#         self.norLabel = self.class_to_idx['NOR']
        
        
        
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx: int) -> Tuple[Image.Image, int, Image.Image]:
#         # Load the image and its corresponding label
#         class_name, filename = self.samples[idx]
#         image = Image.open(os.path.join(self.data_dir, filename))
        
#         original_img = image.copy()
#         not_aug_img = self.not_aug_transform(original_img)
        
#         label = self.class_to_idx[class_name]
#         if self.transform is not None:
#             image = self.transform(image)
#         if self.target_transform is not None:
#             label = self.target_transform(label)
        
#         if hasattr(self, 'logits'):
#             return image, label, not_aug_img, self.logits[idx]
            
#         return image, label, not_aug_img
        

class SequentialGrainSpace(ContinualDataset):

    NAME = 'seq-GrainSpace'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    # split NORMAL into 6 subsets
    N_TASKS = 4
    N_CLASSES = 7
    
    TRANSFORM = transforms.Compose(
            [
             transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.4254, 0.3929, 0.2554],
                                  [0.1206, 0.1556, 0.1535])
            ])

    def get_data_loaders(self):
        transform = self.TRANSFORM

        test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.get_normalization_transform()
        ])
        
        train_path = self.args.dataset_path + '/train'
        val_path = self.args.dataset_path + '/val'
        test_path = self.args.dataset_path + '/test'
        
        train_dataset = GrainDataset(train_path, transform=transform)
        
        if self.args.validation:
            test_dataset = ImageFolder(val_path, transform=test_transform)
        else:
            test_dataset = ImageFolder(test_path, transform=test_transform)
            

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

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
        transform = transforms.Normalize([0.4254, 0.3929, 0.2554],
                                         [0.1206, 0.1556, 0.1535])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize([0.4254, 0.3929, 0.2554],
                                [0.1206, 0.1556, 0.1535])
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
