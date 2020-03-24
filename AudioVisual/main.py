from model import Lipreader
from torch.utils.data import DataLoader
from dataset import AudioVideoDataset
from torch.utils.data.dataloader import DataLoader
import config
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrScheduler
import math
import argparse
import torch
import torch.nn.functional as F
import os
import csv


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


def changeLayers(model, stage, pretrainedDict):
    currentStateDict = model.state_dict()
    pretrainedDict = {k: v for k,
                      v in pretrainedDict.items() if k in currentStateDict}
    currentStateDict.update(pretrainedDict)
    model.load_state_dict(currentStateDict)
    return model


def freezeLayers(model, stage):
    if stage == 2:
        for param in model.convolution3d.parameters():
            param.requires_grad = False
        for param in model.resnet34.parameters():
            param.requires_grad = False
    return model


def loadModel(model, optimizer, scheduler, fileName, stage, updateStage):
    path = os.path.join(
        config.savedModelPath["path"], fileName.split('.')[0], fileName)
    checkpoint = torch.load(path)
    # We want lr = .003 if stage is changed
    if not updateStage:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        model = changeLayers(model, stage, checkpoint["state_dict"])
    model = freezeLayers(model, stage)
    return model, optimizer, scheduler


def saveModel(model, optimizer, scheduler, fileName):
    path = config.savedModelPath["path"]
    if not os.path.exists(path):
        os.mkdir(path)
    dir = fileName.split(".")[0]
    path = os.path.join(path, dir)
    if not os.path.exists(path):
        os.mkdir(path)
    path = os.path.join(path, fileName)
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, path)


def updateLRFunc(epoch):
    if epoch >= 5:
        return math.pow(0.5, (epoch - 5 + 1) / float(5))
    return 1


def trainModel(stage, adam, model, scheduler, dataLoader, criterion):
    model.train()
    for idx, batch in enumerate(dataLoader):
        target, audioInput, videoInput = batch[0], batch[1], batch[2]
        op = model((audioInput, videoInput))
        # loss = criterion(op, target)
        # adam.zero_grad()
        # loss.backward()
        # old_lr = adam.param_groups[0]['lr']
        # adam.step()
        print(op.shape)

    scheduler.step()
    return 1


def validateModel(model, epoch, validationDataLoader, criterion, stage, lr):
    model.eval()
    validationStats = []

    correct = 0
    total = 0
    for idx, batch in enumerate(validationDataLoader):
        target, input = batch[0], batch[1]
        op = model(input.transpose(1, 2))
        correct += criterion(op, target)
        total += len(batch)
        validationStat = {
            "Stage": stage,
            "Epoch": epoch,
            "Batch": idx + 1,
            "validationVideos": input.shape[0],
            "correctValidationOutputs": correct,
        }
        validationStats.append(validationStat)
    saveStatsToCSV(
        validationStats, epoch, "validation", stage)
    print(
        f"Out of {total} videos, {correct} {'was' if correct==1 else 'were'} classified  correctly after epoch {epoch} with lr = {lr} ")


def saveStatsToCSV(data, epoch, mode, stage):
    fileName = f"Epoch{epoch}_{stage}.csv"
    dir = f"Epoch{epoch}_{stage}"
    file = os.path.join(config.savedModelPath["path"], dir, fileName)
    with open(file, 'w') as csvfile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line args')
    parser.add_argument('--load', type=str,
                        help='Name of the file containing the model')
    args = parser.parse_args()
    fileName = args.load

    # default setup
    startEpoch = 1
    stage = 1
    epochs = config.stage["epochs"][stage-1]
    lr = 3e-4
    if fileName is not None:
        if not os.path.exists(os.path.join('.', 'savedModels', fileName.split('.')[0])):
            raise Exception("No such file exists")

        # File is stored as Epoch23_1.pt
        startEpoch = int(fileName.split('_')[0][5:]) + 1
        stage = int(fileName.split('_')[1][0])
        epochs = config.stage["epochs"][stage - 1]

        if startEpoch > epochs:
            if stage == 3:
                print("Succesfully trained all 3 stages")
                exit(0)
            startEpoch = 1
            stage = stage + 1
            epochs = config.stage["epochs"][stage - 1]
            print(f"Updated stage to {stage}")

        path = config.savedModelPath["path"]
        if not os.path.exists(path):
            os.mkdir(path)
        dir = fileName.split(".")[0]
        path = os.path.join(path, dir)

        model = Lipreader()
        adam = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)
        scheduler = lrScheduler.LambdaLR(adam, lr_lambda=[updateLRFunc])

        if os.path.exists(path) and os.path.exists(os.path.join(path, fileName)):
            model, optimizer, scheduler = loadModel(
                model, adam, scheduler, fileName, stage, startEpoch == 1)
            if startEpoch is not 1:
                print(
                    f"Successfully loaded model with last completed epoch as {startEpoch-1}")

        else:
            raise Exception("No such file exixts")
    else:
        model = Lipreader()
        adam = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)
        scheduler = lrScheduler.LambdaLR(adam, lr_lambda=[updateLRFunc])

    trainDataset = AudioVideoDataset("train")
    trainDataLoader = DataLoader(trainDataset, batch_size=config.data["batchSize"],
                                 shuffle=config.data["shuffle"], num_workers=config.data["workers"])
    validationDataset = AudioVideoDataset("val")
    validationDataLoader = DataLoader(validationDataset, batch_size=config.data["batchSize"],
                                      shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    trainCriterion = NLLSequenceLoss()

    validationCriterion = gruValidator

    # model = freezeLayers(model, stage)
    for epoch in range(startEpoch - 1, epochs):
        old_lr = trainModel(1, adam, model, scheduler,
                            trainDataLoader, trainCriterion)
        saveModel(
            model, adam, scheduler, f'Epoch{epoch+1}_{stage}.pt')
        validateModel(model, epoch+1, validationDataLoader,
                      validationCriterion, stage, old_lr)
    print(f"Successfully completed stage {stage} of training")
