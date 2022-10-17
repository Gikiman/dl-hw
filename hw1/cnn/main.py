from genericpath import exists
import torch
import torch.nn as nn
import torch.optim as optim
import data
import models
import os
import argparse
import matplotlib.pyplot as plt

## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model,train_loader, valid_loader, criterion, optimizer, save_dir, num_epochs, model_type):

    def train(model, train_loader,optimizer,criterion):
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
        train_loss, train_acc = train(model, train_loader,optimizer,criterion)
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

    plt.subplot(2, 2, 1)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(train_loss_curve)), train_loss_curve)

    plt.subplot(2, 2, 2)
    plt.title('Valid Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(range(len(valid_loss_curve)), valid_loss_curve)

    plt.subplot(2, 2, 3)
    plt.title('Training Acc Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.plot(range(len(train_acc_curve)), train_acc_curve)
    
    plt.subplot(2, 2, 4)
    plt.title('Valid Acc Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Acc')
    plt.plot(range(len(valid_acc_curve)), valid_acc_curve)

    plt.suptitle("Model {} Result".format(model_type))
    plt.savefig("Result.png")

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

    ## loss function
    criterion = nn.CrossEntropyLoss()

    if args.test:   
        vaild_model(model,valid_loader,criterion,args.model_type)
    else:
        train_model(model,train_loader, valid_loader, criterion, optimizer, args.save_dir, num_epochs,args.model_type)
