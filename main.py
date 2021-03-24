import tensorflow as tf
from dataset import make_dataset
from model import TasNet, TasNetParam

param = TasNetParam(N=500, L=40, H=1000, K=10, C=4, g=1.0, b=0.0)
dataset = make_dataset(param, 10, 100)
model = TasNet.make(param)
model.fit(dataset)
model.save("model.h5")