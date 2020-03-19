from dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import config
from model import Lipreader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrScheduler
import math


def freezeLayers(model, stage):
    if stage == 2:
        for param in model.convolution3d.parameters():
            param.requires_grad = False
        for param in model.resnet34.parameters():
            param.requires_grad = False
    return model


def loadModel(model, path, stage):
    model.load_state_dict(torch.load(path))
    model = freezeLayers(model, stage)
    return model


def saveModel(model, path):
    torch.save(model.state_dict(), PATH)


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
    mode = "train"
    stage = 1

    dataset = VideoDataset("train")
    dataLoader = DataLoader(dataset, batch_size=config.data["batchSize"],
                            shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    model = Lipreader()
    criterion = nn.CrossEntropyLoss()
    adam = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.)
    scheduler = lrScheduler.LambdaLR(adam, lr_lambda=[updateLRFunc])
    if mode == "train":
        model.train()
        model = freezeLayers(model, stage)
        for epoch in range(10):
            trainModel(1, adam, model, scheduler, dataLoader)
    elif mode == "val":
        validateModel()
    else:
        testModel()
