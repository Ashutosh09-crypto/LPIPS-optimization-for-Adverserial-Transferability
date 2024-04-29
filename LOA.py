# LOA (Lpips optimization attack)

import torch
import torch.nn as nn
import numpy as np
from utils import *
import torch.nn.functional as F
import torch_dct as dct
import scipy.stats as st
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
from SIT import *

class Attack(object):
    """
    Base class for all attacks.
    """
    def __init__(self, attack, model, epsilon, targeted, random_start, norm, loss,device=None):
        """
        Initialize the hyperparameters
        Arguments:
            attack (str): the name of attack.
            model (torch.nn.Module): the surrogate model for attack.
            epsilon (float): the perturbation budget.
            targeted (bool): targeted/untargeted attack.
            random_start (bool): whether using random initialization for delta.
            norm (str): the norm of perturbation, l2/linfty.
            loss (str): the loss function.
            device (torch.device): the device for data. If it is None, the device would be same as model
        """
        if norm not in ['l2', 'linfty']:
            raise Exception("Unsupported norm {}".format(norm))
        self.attack = attack
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        # self.device = next(model.parameters()).device if device is None else device
        # self.device = 'cpu'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.loss = self.loss_function(loss)
        self.alpha = epsilon
        self.epoch = 10
        self.decay = 1.0

    
    def forward(self, data, label, **kwargs):
        """
        The general attack procedure
        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)
        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))
            # Calculate the loss
            loss = self.get_loss(logits, label)
            # Calculate the gradients
            grad = self.get_grad(loss, delta)
            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum,decay=self.decay)
            delta = self.update_delta(delta, data, momentum, self.alpha)
        return delta.detach()

    def get_logits(self, x, **kwargs):
        """
        The inference stage, which should be overridden when the attack need to change the models (e.g., ensemble-model attack, ghost, etc.) or the input (e.g. DIM, SIM, etc.)
        """
        return self.model(x)

    def get_loss(self, logits, label):
        """
        The loss calculation, which should be overrideen when the attack change the loss calculation (e.g., ATA, etc.)
        """
        # Calculate the loss
        return -self.loss(logits, label) if self.targeted else self.loss(logits, label)
        

    def get_grad(self, loss, delta, **kwargs):
        """
        The gradient calculation, which should be overridden when the attack need to tune the gradient (e.g., TIM, variance tuning, enhanced momentum, etc.)
        """
        return torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
        # print(loss, delta)
        # return torch.autograd.grad(loss, delta)[0]

    def get_momentum(self, grad, momentum, decay=None, **kwargs):
        """
        The momentum calculation
        """
        return momentum * decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))

    def init_delta(self, data, **kwargs):
        delta = torch.zeros_like(data).to(self.device)
        if self.random_start:
            if self.norm == 'linfty':
                delta.uniform_(-self.epsilon, self.epsilon)
            else:
                delta.normal_(-self.epsilon, self.epsilon)
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=10).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(data).uniform_(0,1).to(self.device)
                delta *= r/n*self.epsilon
            delta = clamp(delta, img_min-data, img_max-data)
        delta.requires_grad = True
        return delta

    def update_delta(self, delta, data, grad, alpha, **kwargs):
        if self.norm == 'linfty':
            delta = torch.clamp(delta + alpha * grad.sign(), -self.epsilon, self.epsilon)
        else:
            grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
            scaled_grad = grad / (grad_norm + 1e-20)
            delta = (delta + scaled_grad * alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.epsilon).view_as(delta)
        delta = clamp(delta, img_min-data, img_max-data)
        return delta


    def loss_function(self, loss):
        """
        Get the loss function
        """
        if loss == 'crossentropy':
            return nn.CrossEntropyLoss()
        else:
            raise Exception("Unsupported loss {}".format(loss))

    def transform(self, data, **kwargs):
        return data

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.forward(*input, **kwargs)
    
class MIFGSM(Attack):
    """
    MI-FGSM Attack
    'Boosting Adversarial Attacks with Momentum (CVPR 2018)'(https://arxiv.org/abs/1710.06081)

    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.
    """
    
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='MI-FGSM', **kwargs):
        super().__init__(attack, model, epsilon, targeted, random_start, norm, loss, device,**kwargs)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

class LOA(MIFGSM):
    """
    LOA Attack
    
    Arguments:
        model (torch.nn.Module): the surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_optim (int): the number of Lpips optimization steps.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3
    """
    
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_copies=20, num_optim=4, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy',device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), attack='LOA', choice = -1,**kwargs):
        super().__init__(model, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack, **kwargs)
        self.num_copies = num_copies
        self.num_optim = num_optim
        self.epsilon = epsilon
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', normalize=True).to(self.device)
        self.alpha = alpha
        self.choice = choice

    def add_noise(self, x, noise=None):
        if noise != None:
               return torch.clip(x + noise, 0, 1)
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)
    
    def init_lpips_delta(self, data, method=None):
      delta = None
      if method == None:
            delta = torch.zeros_like(data).to(self.device)
            delta.uniform_(-self.epsilon, self.epsilon)
            delta.requires_grad = True
      elif method == "SIT":
        data_copy = blocktransform(data)
        # copy_img = tensor_to_image(data_copy[0])
        delta = data_copy - data
      
      return delta
    
    def get_lpips_loss(self, image1, image2):
       res = self.lpips_loss(image1, image2)
       return res
    
    def update_lpips_delta(self, delta, grad):
      delta = torch.clamp(delta + self.alpha * grad.sign(), -self.epsilon, self.epsilon)
      return delta

    def lpips_transform(self, x, mask=None):
        # transformed_image = x.copy()
        transformed_image = x
        
        
        if (self.choice == 1) :
            delta = self.init_lpips_delta(transformed_image, method="SIT")
        else:
            delta = self.init_lpips_delta(transformed_image)

        noised_image = self.add_noise(transformed_image, delta)
        # noised_image = noised_image.view(1,3,1000,1500)

        momentum = 0.
        for _ in range(self.num_optim):
            loss = self.get_lpips_loss(transformed_image, noised_image)
            grad = torch.autograd.grad(loss, delta, retain_graph=True)[0].to(self.device)
            momentum = momentum * self.decay + grad / (grad.abs().mean(dim=(1,2,3), keepdim=True))
            delta = self.update_lpips_delta(delta, momentum)
            noised_image = self.add_noise(noised_image, delta)

        if (self.choice == 2) : # Add the SIT after LPIPS optimization
            sit_delta = self.init_lpips_delta(self, noised_image)
            noised_image = self.add_noise(noised_image, sit_delta)

        return noised_image

    def transform(self, x,**kwargs):
        """
        Scale the input for lpips_transform
        """
        return torch.cat([self.lpips_transform(x) for _ in range(self.num_copies)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return self.loss(logits, label.repeat(self.num_copies))