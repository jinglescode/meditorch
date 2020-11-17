import torch

import numpy as np
import copy
import time
from collections import defaultdict

from .metrics import dice_loss, intersection_over_union
from .utils.plot import metrics_line, normalise_mask


class Trainer(object):

    def __init__(self, model, optimizer=None, scheduler=None, verbose=False):

        super().__init__()
        
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.optimizer = optimizer
        if self.optimizer == None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.scheduler = scheduler
        if self.scheduler == None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)


    def train_model(self, dataloaders, num_epochs=25):
        print("Training model, size:", count_params(self.model))
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10
        epochs_metrics = {
            'train': [],
            'val': []
        }
        best_epoch = {}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))

            since = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                    if self.verbose:
                        for param_group in self.optimizer.param_groups:
                            print("\tlearning rate: {:.2e}".format(param_group['lr']))                    
                else:
                    self.model.eval()   # Set model to evaluate mode

                metrics = defaultdict(float)
                epoch_samples = 0

                for inputs, labels in dataloaders[phase]:
#                     inputs = inputs.to(self.device)
#                     labels = labels.to(self.device)
                    
                    inputs = inputs.to(device=self.device, dtype=torch.float)
                    labels = labels.to(device=self.device, dtype=torch.float)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    epoch_samples += inputs.size(0)

                computed_metrics = compute_metrics(metrics, epoch_samples)
                if self.verbose: 
                    print_metrics(computed_metrics, phase)
                epochs_metrics[phase].append(computed_metrics)
                epoch_loss = metrics['loss'] / epoch_samples

                # deep copy the model
                if phase == 'val':
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(epoch_loss)
                    else:
                        self.scheduler.step()
                        
                    if epoch_loss < best_loss:
                        if self.verbose > 1: 
                            print("\tSaving best model, epoch loss {:4f} < best loss {:4f}".format(epoch_loss, best_loss))
                        best_loss = epoch_loss
                        best_model_wts = copy.deepcopy(self.model.state_dict())

                        best_epoch = {
                            "epoch": epoch,
                            "loss": computed_metrics['loss'],
                            "bce": computed_metrics['bce'],
                            "dice": computed_metrics['dice'],
                            "iou": computed_metrics['iou'],
                        }

            time_elapsed = time.time() - since
            if self.verbose > 1: 
                print('\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                print('-' * 10)

        print("Best epoch:", best_epoch)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        metrics_line(epochs_metrics)
        
        return best_epoch

    def predict(self, X, Y=None, threshold=0.5):
        self.model.eval()
        inputs = X.to(device=self.device, dtype=torch.float)
        outputs = self.model(inputs)

        pred = outputs.data.cpu().numpy()

        pred = normalise_mask(pred, threshold)
        
        if Y is not None:
            metrics = defaultdict(float)
            labels = Y.to(device=self.device, dtype=torch.float)
            loss = calc_loss(outputs, labels, metrics)
            
            epoch_samples = X.size(0)
            computed_metrics = compute_metrics(metrics, epoch_samples)
            print_metrics(computed_metrics, "test")
            
            return pred, computed_metrics

        return pred


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    pred_binary = normalise_mask(pred.detach().cpu().numpy())
    iou = intersection_over_union(target.detach().cpu().numpy(), pred_binary)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['iou'] += iou * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss

def compute_metrics(metrics, epoch_samples):
    computed_metrics = {}
    for k in metrics.keys():
        computed_metrics[k] = metrics[k] / epoch_samples
    return computed_metrics

def print_metrics(computed_metrics, phase):
    outputs = []
    for k in computed_metrics.keys():
        outputs.append("{}:{:4f}".format(k, computed_metrics[k]))
    print("\t{}-> {}".format(phase.ljust(5), "|".join(outputs)))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# from collections import defaultdict
# import copy
# import time
# import bz2
# import pickle
# import torch

# from .metrics import dice_loss, intersection_over_union
# from .utils.plot import metrics_line, normalise_mask


# class Trainer(object):

#     def __init__(self, model, optimizer=None, scheduler=None):

#         super().__init__()

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.model = model.to(self.device)

#         self.optimizer = optimizer
#         if self.optimizer == None:
#             self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

#         self.scheduler = scheduler
#         if self.scheduler == None:
#             self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

#         self.best_epoch_info = {}

#     def train_model(self, dataloaders, num_epochs=25, verbose=False):

#         time_fit_start = time.time()

#         # best_model_wts = copy.deepcopy(self.model.state_dict())
#         # best_loss = 1e10

#         epochs_metrics = {
#             'train': [],
#             'val': []
#         }
#         self.best_epoch_info = {
#             'model_wts':copy.deepcopy(self.model.state_dict()),
#             'loss':1e10,
#             'metric':1e10
#         }

#         for epoch in range(num_epochs):
#             if verbose or epoch%(num_epochs/5)==0:
#                 print('Epoch {}/{}'.format(epoch+1, num_epochs))

#             since = time.time()

#             # Each epoch has a training and validation phase
#             for phase in ['train', 'val']:
#                 if phase == 'train':
#                     for param_group in self.optimizer.param_groups:
#                         if verbose:
#                             print("\tlearning rate: {:.2e}".format(param_group['lr']))

#                     self.model.train()  # Set model to training mode
#                 else:
#                     self.model.eval()   # Set model to evaluate mode

#                 metrics = defaultdict(float)
#                 epoch_samples = 0

#                 for inputs, labels in dataloaders[phase]:
#                     inputs = inputs.to(self.device)
#                     labels = labels.to(self.device)

#                     # zero the parameter gradients
#                     self.optimizer.zero_grad()

#                     # forward
#                     with torch.set_grad_enabled(phase == 'train'):
#                         outputs = self.model(inputs)
#                         loss = calc_loss(outputs, labels, metrics)

#                         # backward + step optimizer only if in training phase
#                         if phase == 'train':
#                             loss.backward()
#                             self.optimizer.step()

#                     # statistics
#                     epoch_samples += inputs.size(0)

#                 computed_metrics = compute_metrics(metrics, epoch_samples)
#                 if verbose:
#                     print_metrics(computed_metrics, phase)
#                 epochs_metrics[phase].append(computed_metrics)
#                 epoch_loss = metrics['loss'] / epoch_samples

#                 if phase == 'train':
#                     self.scheduler.step()

#                 # deep copy the model
#                 if phase == 'val': # and epoch_loss < self.best_epoch_info['loss']:
#                     if computed_metrics['bce'] < self.best_epoch_info['metric'] or epoch_loss < self.best_epoch_info['loss']:
#                         # print("\tSaving best model, epoch loss {:4f} < best loss {:4f}".format(epoch_loss, self.best_epoch_info['loss']))
#                         if verbose:
#                             print('\tSaving best model')
#                         # best_loss = epoch_loss
#                         # best_model_wts = copy.deepcopy(self.model.state_dict())
#                         self.best_epoch_info = {
#                             'epoch':epoch,
#                             'metrics':computed_metrics,
#                             'loss':epoch_loss,
#                             'metric':computed_metrics['bce'],
#                             'model_wts':copy.deepcopy(self.model.state_dict())
#                         }

#             time_elapsed = time.time() - since
#             if verbose:
#                 print('\t{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#                 print('-' * 10)

#         time_elapsed = time.time() - time_fit_start
#         print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#         print('Training summary, best epoch info:')
#         print('Epoch', self.best_epoch_info['epoch'])
#         print('Val loss: {:4f}'.format(self.best_epoch_info['loss']))
#         print('Metrics:')
#         for metric in self.best_epoch_info['metrics']:
#             print('\t' + metric, self.best_epoch_info['metrics'][metric])

#         # load best model weights
#         self.model.load_state_dict(self.best_epoch_info['model_wts'])

#         metrics_line(epochs_metrics)

#     def predict(self, X, threshold=0.5):
#         self.model.eval()
#         inputs = X.to(self.device)
#         pred = self.model(inputs)

#         pred = pred.data.cpu().numpy()

#         pred = normalise_mask(pred, threshold)

#         return pred

#     def save_model(self, bz2_filename):
#         bz2file = bz2.BZ2File(bz2_filename, 'w')
#         pickle.dump(self.best_epoch_info, bz2file)
#         bz2file.close()
#         print('Model saved:', bz2_filename)

#     def load_model(self, bz2_filename):
#         bz2file = bz2.BZ2File(bz2_filename)
#         best_epoch_info = pickle.load(bz2file)
#         self.model.load_state_dict(best_epoch_info['model_wts'])
#         self.best_epoch_info = best_epoch_info
#         print('Model loaded:', bz2_filename)


# def calc_loss(pred, target, metrics, bce_weight=0.5):
#     bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)

#     pred = torch.sigmoid(pred)
#     dice = dice_loss(pred, target)

#     pred_binary = normalise_mask(pred.detach().cpu().numpy())
#     iou = intersection_over_union(target.detach().cpu().numpy(), pred_binary)

#     loss = bce * bce_weight + dice * (1 - bce_weight)

#     metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
#     metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
#     metrics['iou'] += iou * target.size(0)
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

#     return loss

# def compute_metrics(metrics, epoch_samples):
#     computed_metrics = {}
#     for k in metrics.keys():
#         computed_metrics[k] = metrics[k] / epoch_samples
#     return computed_metrics

# def print_metrics(computed_metrics, phase):
#     outputs = []
#     for k in computed_metrics.keys():
#         outputs.append("{}:{:4f}".format(k, computed_metrics[k]))

#     print("\t{}-> {}".format(phase.ljust(5), "|".join(outputs)))
