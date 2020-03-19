from dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import config
from model import Lipreader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrScheduler
import math
import argparse
import torch


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


if __name__ == "__main__":
    model = Lipreader()
    adam = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)
    scheduler = lrScheduler.LambdaLR(
        adam, lr_lambda=[updateLRFunc])

    parser = argparse.ArgumentParser(description='Process command line args')
    parser.add_argument('--load', type=str,
                        help='Name of the file containing the model')
    args = parser.parse_args()
    fileName = args.load
    startEpoch = 1
    stage = 1
    epochs = 30
    mode = "train"
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

    criterion = nn.CrossEntropyLoss()
    if mode == "train":
        model.train()
        model = freezeLayers(model, stage)
        for epoch in range(startEpoch - 1, epochs):
            trainModel(1, adam, model, scheduler, trainDataLoader)
            saveModel(
                model, adam, scheduler, f'Epoch{epoch+1}_{stage}.pt')
    elif mode == "val":
        validateModel()
    else:
        testModel()
