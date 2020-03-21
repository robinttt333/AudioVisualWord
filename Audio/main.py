from dataset import AudioDataset
from torch.utils.data.dataloader import DataLoader
import config
from model import Lipreader


trainDataset = AudioDataset("train")
trainDataLoader = DataLoader(trainDataset, batch_size=config.data["batchSize"],
                             shuffle=config.data["shuffle"], num_workers=config.data["workers"])
model = Lipreader()
for idx, batch in enumerate(trainDataLoader):
    target, input = batch[0], batch[1]
    op = model(input)
