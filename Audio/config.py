import os
data = {
    "path": "../data",
    "processedPath": "./data",
    "batchSize": 36,
    "shuffle": True,
    "workers": 4,
}

stage = {
    "epochs": [30, 5, 30]
}

savedModelPath = {
    "path": os.path.join(os.getcwd(), "savedModels")
}
