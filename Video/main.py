from dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import toml
import config
from model import Lipreader


def trainModel():
    model = Lipreader()
    for idx, batch in enumerate(dataLoader):
        target, input = batch[0], batch[1]
        op = model(input.transpose(1, 2))


if __name__ == "__main__":
    dataset = VideoDataset("train")
    dataLoader = DataLoader(dataset, batch_size=config.data["batchSize"],
                            shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    mode = "train"
    if mode == "train":
        trainModel()
    elif mode == "val":
        validateModel()
    else:
        testModel()
