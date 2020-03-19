from dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import config
from model import Lipreader, TemporalCNN, GRU
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrScheduler
import math
import argparse
import torch
import torch.nn.functional as F


def temporalCNNValidator(outputs, labels):
    """ Here we get a batchSize * total labels(500) outputs shape and batchSize shaped labels.eg - 
        Consider the word afternoon [ 10 * 500 ] -----> [1]
        It maps from a 10 * 500 tensor(matrix) to a single label.  
    """
    """Calculate the max for each batch out of the 500 possible outputs. In return we get the 
    index of the word along with its actual probability which is of little use to us"""
    maxvalues, maxindices = torch.max(outputs, 1)
    count = 0
    for i in range(0, labels.size(0)):
        if maxindices[i] == labels[i]:
            count += 1

    return count  # return the number of correct predictions in the batch


def gruValidator(outputs, labels):
    """The input is of the form batchsize * frames * total labels. So we need to get the correct
    label corresponding to each frame.We first take avg across all 29 frames in a video giving us back a vector with
    dimensions batchSize * total labels and then the process is similar to what we did in the function above."""

    # same as taking avg as sum/29 for all labels
    outputsTransformed = torch.sum(outputs, 1)
    return temporalCNNValidator(outputsTransformed, labels)


class NLLSequenceLoss(nn.Module):

    def __init__(self):
        super(NLLSequenceLoss, self).__init__()
        self.criterion = nn.NLLLoss()

    def forward(self, input, target):
        loss = 0.0
        input = F.softmax(input, dim=2)
        transposed = input.transpose(0, 1).contiguous()
        for i in range(0, 29):
            loss += self.criterion(transposed[i], target)

        return loss


def freezeLayers(model, stage):
    if stage == 2:
        for param in model.convolution3d.parameters():
            param.requires_grad = False
        for param in model.resnet34.parameters():
            param.requires_grad = False
    return model


def loadModel(model, optimizer, scheduler, path, stage):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model = freezeLayers(model, stage)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler


def saveModel(model, optimizer, scheduler, path):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)


def updateLRFunc(epoch):
    if epoch >= 5:
        return math.pow(0.5, (epoch - 5 + 1) / float(5))
    return 1


def trainModel(stage, adam, model, scheduler, dataLoader):
    for idx, batch in enumerate(dataLoader):
        target, input = batch[0], batch[1]
        op = model(input.transpose(1, 2))
        loss = criterion(op, target)
        adam.zero_grad()
        loss.backward()
        adam.step()

    scheduler.step()


def validateModel(model, epoch):
    model.eval()
    validationDataset = VideoDataset("val")
    validationDataLoader = DataLoader(validationDataset, batch_size=config.data["batchSize"],
                                      shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    criterion = temporalCNNValidator if isinstance(
        model.Backend, TemporalCNN) else gruValidator

    correct = 0
    total = 0
    for idx, batch in enumerate(validationDataLoader):
        target, input = batch[0], batch[1]
        op = model(input.transpose(1, 2))
        correct += criterion(op, target)
        total += len(batch)
    print(
        f'Out of {total} videos,{correct} were classified correctly after epoch {epoch+1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line args')
    parser.add_argument('--load', type=str,
                        help='Name of the file containing the model')
    args = parser.parse_args()
    fileName = args.load

    model = Lipreader(
        int(fileName.split('_')[1][0]) if fileName is not None else 1)
    adam = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)
    scheduler = lrScheduler.LambdaLR(
        adam, lr_lambda=[updateLRFunc])

    startEpoch = 1
    stage = 1
    epochs = 30
    lr = 3e-4
    if fileName is not None:
        model, optimizer, scheduler = loadModel(
            model, adam, scheduler, fileName, stage)
        # File is stored as Epoch23_1.pt
        startEpoch = int(fileName.split('_')[0][5:]) + 1
        stage = int(fileName.split('_')[1][0])
        epochs = config.stage["epochs"][stage - 1]

    trainDataset = VideoDataset("train")
    trainDataLoader = DataLoader(trainDataset, batch_size=config.data["batchSize"],
                                 shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    criterion = nn.CrossEntropyLoss() if isinstance(
        model.Backend, TemporalCNN) else NLLSequenceLoss()

    model.train()
    model = freezeLayers(model, stage)
    for epoch in range(startEpoch - 1, epochs + 1):
        trainModel(1, adam, model, scheduler, trainDataLoader)
        saveModel(
            model, adam, scheduler, f'Epoch{epoch+1}_{stage}.pt')

        validateModel(model, epoch)
