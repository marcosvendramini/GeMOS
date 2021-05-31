#Import
import numpy as np
import torch
import itertools

from torch import nn
from torch.utils import data
from torch.backends import cudnn

from torchvision import models
from torchvision import transforms
from torchvision import datasets

from sklearn import metrics
from sklearn import mixture

from matplotlib import pyplot as plt

import random

cudnn.benchmark = True

args = {
    'batch_size': 200,    # Batch Size.
    'num_workers': 4,     # Dataloader threads.
    'num_classes': 10,    # Number of KKCs.
    'num_components': 4,  # Number of Components.
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

def train_gemos(mnist_train_loader, net):
    
    with torch.no_grad():
        
        # Setting network for evaluation mode (not computing gradients).
        net.eval()

        # Lists for output features.
        cls_list = [[] for c in range(args['num_classes'])]

        # Iterating over batches.
        for i, batch_data in enumerate(mnist_train_loader):

            # Obtaining images, labels and paths for batch.
            inps, labs = batch_data

            # Casting to cuda variables.
            inps = inps.cuda()

            # Forwarding.
            outs = net(inps)

            # Obtaining predictions.
            prds = outs.data.max(dim=1)[1].cpu().numpy()

            for j in range(prds.shape[0]):

                prds_cls = prds[j]
                labs_cls = labs[j].detach().cpu().item()

                if prds_cls == labs_cls:

                    cls_list[labs_cls].append(outs[j].detach().cpu().numpy().ravel())

        model_list = []

        for c in range(args['num_classes']):

            print('Training model for class %d...' % (c))
            model_list.append(train_gmm(np.asarray(cls_list[c]), args['num_components']))

        return model_list
    

def test_gemos(test_loader, net, model_list):
    
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

            if inps.size(1) > 1:
                inps = inps[:, 0, :, :].unsqueeze(1)

            # Casting to cuda variables.
            inps = inps.cuda()
            labs = labs.cuda()

            rand = random.randint(0, 199)

            # Forwarding.
            outs = net(inps)

            # Obtaining predictions.
            prds = outs.data.max(dim=1)[1].cpu().numpy()

            for j in range(prds.shape[0]):

                prds_cls = prds[j]

                outs_cls = outs[j].detach().cpu().numpy().ravel()

                scr = model_list[prds_cls].score(np.expand_dims(outs_cls, 0))

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

# Setting root dirs.
dir_omniglot = './OMNIGLOT/'
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

#Create Model
model = models.resnet18().cuda()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).cuda()
model.fc = nn.Linear(in_features=512, out_features=args['num_classes'], bias=True).cuda()

#Load model
state = torch.load('./MNIST_Omniglot_GMM_4comp_model.pth')
model.load_state_dict(state)
print(model)

model_list = train_gemos(mnist_train_loader, model)

print('Processing MNIST...')
mnist_scr_list, mnist_prd_list, mnist_lab_list, mnist_out_list, mnist_inps, mnist_inps_prd, mnist_inps_scr, mnist_inps_lab = test_gemos(mnist_test_loader, model, model_list)

print('Processing OMNIGLOT...')
omniglot_scr_list, omniglot_prd_list, omniglot_lab_list, omniglot_out_list, omniglot_inps, omniglot_inps_prd, omniglot_inps_scr, omniglot_inps_lab = test_gemos(omniglot_test_loader, model, model_list)
