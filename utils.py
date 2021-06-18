#Import
import numpy as np
import torch
import time
import os
import joblib
import itertools

from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils import data
from torch.backends import cudnn

from torchvision import models
from torchvision import transforms
from torchvision import datasets

from sklearn import metrics
from sklearn import mixture

from matplotlib import pyplot as plt

import random

from models import *

cudnn.benchmark = True

args = {
    'num_workers': 4,     # Dataloader threads.
    'momentum': 0.9,      # Momentum.
    'num_classes': 10,    # Number of KKCs.
    'num_components': 4,  # Number of Components.
    'lr_mnist': 1e-3,           # Learning rate.
    'weight_decay_mnist': 5e-5, # Weight decay.
    'epoch_num_mnist': 100,     # Number of epochs.
    'batch_size': 200,    # Batch Size.
    'lr_cifar': 0.1,           # Learning rate.
    'weight_decay_cifar': 5e-4, # Weight decay.
    'epoch_num_cifar': 350,     # Number of epochs.
}

class Negative(object):
    
    def __init__(self):
        pass

    def __call__(self, image):

        image = image * -1.0

        return image

def train_gmm(cls_list, n_components):
    
    model = mixture.GaussianMixture(n_components=n_components, random_state=12345)
    
    model.fit(cls_list)
    
    return model

def train_gemos(train_loader, net, d_in):
    
    with torch.no_grad():
        
        # Setting network for evaluation mode (not computing gradients).
        net.eval()

        # Lists for output features.
        cls_list = [[] for c in range(args['num_classes'])]

        # Iterating over batches.
        for i, batch_data in enumerate(train_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = batch_data

            # Casting to cuda variables.
            inps = inps.cuda()
            
            # Forwarding.
            if d_in == 'MNIST':
                
                outs = net(inps)
                
            elif d_in == 'CIFAR10':
                
                outs, fv1, fv2, fv3, fv4 = net(inps, feats=True)
                
                features = torch.cat([outs, fv1, fv2, fv3, fv4], dim=1)

            # Obtaining predictions.
            prds = outs.data.max(dim=1)[1].cpu().numpy()

            for j in range(prds.shape[0]):

                prds_cls = prds[j]
                labs_cls = labs[j].detach().cpu().item()

                if prds_cls == labs_cls:
                    
                    if d_in == 'MNIST':

                        cls_list[labs_cls].append(outs[j].detach().cpu().numpy().ravel())
                
                    elif d_in == 'CIFAR10':
                
                        cls_list[labs_cls].append(features[j].detach().cpu().numpy().ravel())

        model_list = []

        for c in range(args['num_classes']):

            print('Training model for class %d...' % (c))
            model_list.append(train_gmm(np.asarray(cls_list[c]), args['num_components']))

        return model_list
    

def test_gemos(test_loader, net, model_list, d_in):
    
    with torch.no_grad():
        
        # Setting network for evaluation mode (not computing gradients).
        net.eval()

        # Lists for losses and metrics.
        scr_list = []
        prd_list = []
        lab_list = []
        out_list = []
        inps_list = []
        inps_prd = []
        inps_scr = []
        inps_lab = []

        # Iterating over batches.
        for i, batch_data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = batch_data
            
            if d_in == 'MNIST':
                if inps.size(1) > 1:
                    inps = inps[:, 0, :, :].unsqueeze(1)

            # Casting to cuda variables.
            inps = inps.cuda()
            labs = labs.cuda()

            rand = random.randint(0, 199)
            
            # Forwarding.
            if d_in == 'MNIST':
                
                outs = net(inps)
                
            elif d_in == 'CIFAR10':
                
                outs, fv1, fv2, fv3, fv4 = net(inps, feats=True)
                
                features = torch.cat([outs, fv1, fv2, fv3, fv4], dim=1)

            # Obtaining predictions.
            prds = outs.data.max(dim=1)[1].cpu().numpy()
            scr_temp = np.zeros(prds.shape[0], dtype=np.float32)
                    
            
            for j in range(prds.shape[0]):

                prds_cls = prds[j]
                outs_cls = outs[j].detach().cpu().numpy().ravel()
                
                if d_in == 'MNIST':
                    scr = model_list[prds_cls].score(np.expand_dims(outs_cls, 0))
                
                elif d_in == 'CIFAR10':
                    features_cls = features[prds == prds_cls].detach().cpu().numpy()

                    if features_cls.shape[0] > 0:
                        scr = model_list[prds_cls].score_samples(features_cls)

                scr_list.append(scr)
                out_list.append(outs_cls)

                if j == rand:
                    inps_list.append(inps[j])
                    inps_prd.append(prds[j])
                    inps_scr.append(scr)
                    inps_lab.append(labs[j])

            # Updating lists.
            prd_list.extend(prds.tolist())
            lab_list.extend(labs.detach().cpu().numpy().tolist())

        return scr_list, prd_list, lab_list, out_list, inps_list, inps_prd, inps_scr, inps_lab
    
def train(train_loader, net, criterion, optimizer, epoch):

    tic = time.time()
    
    # Setting network for training mode.
    net.train()

    # Lists for losses and metrics.
    train_loss = []
    
    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):

        # Obtaining images, labels and paths for batch.
        inps, labs = batch_data
        
        # Casting to cuda variables.
        inps = inps.cuda()
        labs = labs.cuda()
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)

        # Computing loss.
        loss = criterion(outs, labs)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()
        
        # Updating lists.
        train_loss.append(loss.data.item())
    
    toc = time.time()
    
    train_loss = np.asarray(train_loss)
    
    # Printing training epoch loss and metrics.
    print('--------------------------------------------------------------------')
    print('[epoch %d], [train loss %.4f +/- %.4f], [training time %.2f]' % (
        epoch, train_loss.mean(), train_loss.std(), (toc - tic)))
    print('--------------------------------------------------------------------')
    
    return train_loss.mean(), train_loss.std()
    
def test(test_loader, net, criterion, epoch):

    with torch.no_grad():
        
        tic = time.time()

        # Setting network for evaluation mode (not computing gradients).
        net.eval()

        # Lists for losses and metrics.
        test_loss = []
        prd_list = []
        lab_list = []

        # Iterating over batches.
        for i, batch_data in enumerate(test_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = batch_data

            # Casting to cuda variables.
            inps = inps.cuda()
            labs = labs.cuda()

            # Forwarding.
            outs = net(inps)

            # Computing loss.
            loss = criterion(outs, labs)

            # Obtaining predictions.
            prds = outs.data.max(dim=1)[1].cpu().numpy()

            # Updating lists.
            test_loss.append(loss.data.item())
            prd_list.append(prds)
            lab_list.append(labs.detach().cpu().numpy())

        toc = time.time()

        # Computing accuracy.
        acc = metrics.accuracy_score(np.asarray(lab_list).ravel(),
                                     np.asarray(prd_list).ravel())

        test_loss = np.asarray(test_loss)

        # Printing test epoch loss and metrics.
        print('--------------------------------------------------------------------')
        print('[epoch %d], [test loss %.4f +/- %.4f], [acc %.4f], [testing time %.2f]' % (
            epoch, test_loss.mean(), test_loss.std(), acc, (toc - tic)))
        print('--------------------------------------------------------------------')

        return test_loss.mean(), test_loss.std()

def mnist_train_test_loader():
    # Setting root dirs.
    dir_omniglot = './Omniglot/'
    dir_mnist = './MNIST/'

    common_transform = transforms.Compose([transforms.Resize(28),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)),])

    ood_transform = transforms.Compose([transforms.Resize(28),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    Negative(),])

    # Setting Closed Set Dataset.
    mnist_train = datasets.MNIST(
    dir_mnist,
    train=True,
    download=True,
    transform=common_transform
    )
    mnist_test = datasets.MNIST(
    dir_mnist,
    train=False,
    download=True,
    transform=common_transform
    )

    mnist_train_loader = data.DataLoader(
    mnist_train,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=True
    )
    
    mnist_test_loader = data.DataLoader(
    mnist_test,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=False
    )

    # Setting Open Set Dataset.
    omniglot_test = datasets.Omniglot(
    dir_omniglot,
    download=True,
    transform=ood_transform
    )

    omniglot_test_loader = data.DataLoader(
    omniglot_test,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=False
    )
    return mnist_train_loader, mnist_test_loader, omniglot_test_loader

def cifar10_train_test_loader():
    # Setting root dirs.
    dir_cifar10 = './CIFAR10/'
    dir_imagenet = './Tiny_ImageNet_Resize/Imagenet_resize/'
    
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    common_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    # Setting Closed Set Dataset.
    cifar10_train = datasets.CIFAR10(
    dir_cifar10,
    train=True,
    download=True,
    transform=transform_train
    )
    
    cifar10_test = datasets.CIFAR10(
    dir_cifar10,
    train=False,
    download=True,
    transform=common_transform
    )

    cifar10_train_loader = data.DataLoader(
    cifar10_train,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=True
    )
    
    cifar10_test_loader = data.DataLoader(
    cifar10_test,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=False
    )

    # Setting Open Set Dataset.
    imagenet_test = datasets.ImageFolder(
    dir_imagenet,
    transform=common_transform
    )
    
    imagenet_loader = data.DataLoader(
    imagenet_test,
    batch_size=args['batch_size'],
    num_workers=args['num_workers'],
    shuffle=False
    )
    return cifar10_train_loader, cifar10_test_loader, imagenet_loader

def mnist_model():
    #Create Model
    model = models.resnet18(pretrained=True, progress=False).cuda()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
    model.fc = nn.Linear(in_features=512, out_features=args['num_classes'], bias=True).cuda()
    return model

def cifar10_model():
    model = torch.nn.DataParallel(DenseNet121(),device_ids=[0])
    return model

def start(d_in):
    if d_in == 'MNIST':
        train_loader, test_loader, out_test_loader = mnist_train_test_loader()
        model = mnist_model()
        d_out = 'OMNIGLOT'
    elif d_in == 'CIFAR10':
        train_loader, test_loader, out_test_loader = cifar10_train_test_loader()
        model = cifar10_model()
        d_out = 'TINY_IMAGENET'
    else:
        print("D_in not found")
        return
    
    if d_in == 'MNIST':
        # Defining optimizer.
        optimizer = optim.Adam(model.parameters(),
                       lr=args['lr_mnist'],
                       betas=(args['momentum'], 0.999),
                       weight_decay=args['weight_decay_mnist'])
        # Defining scheduler.
        scheduler = optim.lr_scheduler.StepLR(optimizer, 40, 0.2)
        epoch_num = args['epoch_num_mnist']
    elif d_in == 'CIFAR10':
        # Defining optimizer.
        optimizer = optim.SGD(model.parameters(), lr=args['lr_cifar'],
                      momentum=args['momentum'], weight_decay=args['weight_decay_cifar'])
        # Defining scheduler.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
        epoch_num = args['epoch_num_cifar']
    # Defining loss.
    criterion = nn.CrossEntropyLoss().cuda()

    # Iterating over epochs.
    
    for epoch in range(1, epoch_num + 1):
    
        # Training function.
        train(train_loader, model, criterion, optimizer, epoch)
    
        # Computing test loss and metrics.
        test(test_loader, model, criterion, epoch)
    
        scheduler.step()
    
    print('Saving CNN...')

    root_model = './cls_models/'
    model_path = os.path.join(root_model, '%s_comp_model.pth' % (d_in))

    torch.save(model.state_dict(), model_path)
    
    model_list = train_gemos(train_loader, model, d_in)

    root_model = './gen_models/'
    for i, m in enumerate(model_list):
    
        model_path = os.path.join(root_model, '%s_%dcomp_model_class.pkl' % (d_in,i))

        joblib.dump(m, model_path)

    print('Processing %s...' % (d_in))
    in_scr_list, in_prd_list, in_lab_list, in_out_list, in_inps, in_inps_prd, in_inps_scr, in_inps_lab = test_gemos(test_loader, model, model_list, d_in)

    print('Processing %s...' % (d_out))
    out_scr_list, out_prd_list, out_lab_list, out_out_list, out_inps, out_inps_prd, out_inps_scr, out_inps_lab = test_gemos(out_test_loader, model, model_list, d_in)
    
    return model, model_list, in_inps, in_inps_lab
        
def evaluate(model, model_list, tr, inp, d_in):
    
    model.eval()

    if d_in == 'MNIST':
        out = model(inp.unsqueeze_(0))
    elif d_in == 'CIFAR10':
        out, fv1, fv2, fv3, fv4 = model(inp.unsqueeze_(0), feats=True)
        features = torch.cat([out, fv1, fv2, fv3, fv4], dim=1)

    prd = out.data.max(dim=1)[1].cpu().numpy()
    if d_in == 'MNIST':
        out_cls = out[0].detach().cpu().numpy().ravel()
        scr = model_list[prd[0]].score(np.expand_dims(out_cls, 0))
    elif d_in == 'CIFAR10':
        features_cls = features.detach().cpu().numpy()
        if features_cls.shape[0] > 0:
            scr = model_list[prd[0]].score(features_cls)

    if scr < tr:
        return 'UNK'
    else:
        return prd[0]

