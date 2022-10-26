from genericpath import exists
import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import argparse
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
 
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
 
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
 
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
 
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
 
    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def train_model(model,train_loader, valid_loader, criterion, optimizer, scheduler, save_dir, num_epochs, model_type):

    def train(model, train_loader,optimizer, scheduler,criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            scheduler.step()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.double() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader,criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.double() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    train_loss_curve,train_acc_curve,valid_loss_curve,valid_acc_curve = [],[],[],[]
    if not exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader,optimizer, scheduler,criterion)
        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        print("Model {} training: {:.4f}, {:.4f}".format(model_type,train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader,criterion)
        valid_loss_curve.append(train_loss)
        valid_acc_curve.append(train_acc)
        print("Model {} validation: {:.4f}, {:.4f}".format(model_type,valid_loss, valid_acc))
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, os.path.join(save_dir, 'best_model_{}.pt'.format(model_type)))
    print("Best accuracy: {}".format(best_acc))

    plt.title('Model {} Training Loss'.format(model_type))
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(train_loss_curve)), train_loss_curve)
    plt.savefig('Model_{}_Training_Loss.png'.format(model_type))
    plt.clf()

    plt.title('Valid Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(valid_loss_curve)), valid_loss_curve)
    plt.savefig('Model_{}_Valid_Loss.png'.format(model_type))
    plt.clf()

    plt.title('Training Acc Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.plot(range(len(train_acc_curve)), train_acc_curve)
    plt.savefig('Model_{}_Training_Acc.png'.format(model_type))
    plt.clf()
    
    plt.title('Valid Acc Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.plot(range(len(valid_acc_curve)), valid_acc_curve)
    plt.savefig('Model_{}_Valid_Acc.png'.format(model_type))
    plt.clf()

def vaild_model(model,valid_loader,criterion,model_type):
    model.train(False)
    total_loss = 0.0
    total_correct = 0
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predictions = torch.max(outputs, 1)
        total_loss += loss.item() * inputs.size(0)
        total_correct += torch.sum(predictions == labels.data)
    epoch_loss = total_loss / len(valid_loader.dataset)
    epoch_acc = total_correct.double() / len(valid_loader.dataset)
    print("Model {} validation: {:.4f}, {:.4f}".format(model_type,epoch_loss, epoch_acc))
    #return epoch_loss, epoch_acc.item()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='hw1')
    parser.add_argument('--test',type=bool,default=False)
    parser.add_argument('--model_type',type=str,default='B',choices=['A','B'])
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
    args = parser.parse_args()

    ## about model
    num_classes = 10

    ## about data
    data_dir = args.data_dir ## You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    ## about training
    num_epochs = 100
    lr = 0.001

    if args.test:
        model = torch.load(os.path.join(args.save_dir, 'best_model_{}.pt'.format(args.model_type)))
    else:
        if args.model_type =='A':
            model = models.model_A(num_classes=num_classes)
        else:
            model = models.model_B(num_classes=num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir,input_size=input_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler_steplr = StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

    ## loss function
    criterion = nn.CrossEntropyLoss()

    if args.test:   
        vaild_model(model,valid_loader,criterion,args.model_type)
    else:
        train_model(model,train_loader, valid_loader, criterion, optimizer,scheduler_warmup,args.save_dir, num_epochs,args.model_type)
