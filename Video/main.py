from dataset import VideoDataset
from torch.utils.data.dataloader import DataLoader
import toml
import config
if __name__ == "__main__":
    dataset = VideoDataset("train")
    dataLoader = DataLoader(dataset, batch_size=config.data["batchSize"],
                            shuffle=config.data["shuffle"], num_workers=config.data["workers"])

    for idx, batch in enumerate(dataLoader):
        target, input = batch[0], batch[1]
