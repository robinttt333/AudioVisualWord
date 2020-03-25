import os
data = {
    "processedPathAudio": "../Audio/data",
    "processedPathVideo": "../Video/data",
    "batchSize": 36,
    "shuffle": True,
    "workers": 4,
}

stage = {
    "epochs": [5, 30]
}

savedModelPath = {
    "path": os.path.join(os.getcwd(), "savedModels")
}
