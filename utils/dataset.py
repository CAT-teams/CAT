import logging
from json import dump
from os.path import join

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import Models

__DATASETS_DEFAULT_PATH = './data/'


def loadModelNames():
    return [name for (name, obj) in Models.__dict__.items() if hasattr(obj, '__call__')]


def loadDatasets():
    return dict(cifar10=10, cifar100=100, imagenet=1000)


def saveArgsToJSON(args):
    # save args to JSON
    args.jsonPath = '{}/args.txt'.format(args.save)
    with open(args.jsonPath, 'w') as f:
        dump(vars(args), f, indent=4, sort_keys=True)


def get_dataset(name, train, transform, target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
    root = datasets_path
    if name == 'cifar10':
        cifar_ = datasets.CIFAR10(root=root, train=train, transform=transform, target_transform=target_transform,
                                  download=download)
        return cifar_

    elif name == 'cifar100':
        cifar_ = datasets.CIFAR100(root=root, train=train, transform=transform, target_transform=target_transform,
                                   download=download)
        return cifar_

    elif name == 'imagenet':
        if train:
            root = join(root, 'train')
        else:
            root = join(root, 'val')
        return datasets.ImageFolder(root=root, transform=transform, target_transform=target_transform)


def get_transform(args):
    if args.dataset == 'imagenet':
        resize = 256 if args.model != 'inception_v3' else 299
        crop_size = 224 if args.model != 'inception_v3' else 299
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resize),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
    else:  # cifar
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform_train, transform_test


def load_data(args, logger):
    # init transforms
    logger.info('==> Preparing data..')
    transform_train, transform_test = get_transform(args)

    transform = {'train': transform_train, 'test': transform_test}

    train_data = get_dataset(args.dataset, train=True, transform=transform['train'], datasets_path=args.data)
    test_data = get_dataset(args.dataset, train=False, transform=transform['test'], datasets_path=args.data)

    testLoader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    trainLoader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    statsBatchSize = args.batch / len(args.gpu)
    data_len = 50000 if args.dataset == 'imagenet' else 10000
    rndIdx = 0  # random.randint(0, data_len - statsBatchSize)
    sample = SubsetRandomSampler(np.linspace(rndIdx, rndIdx + statsBatchSize, statsBatchSize + 1, dtype=np.int)[:-1])

    statsLoader = torch.utils.data.DataLoader(train_data, batch_size=int(args.batch / len(args.gpu)), shuffle=False,
                                              num_workers=args.workers, sampler=sample)

    return testLoader, trainLoader, statsLoader


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super(self.__class__, self).__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
