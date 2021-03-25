import tensorflow as tf
from os import path, listdir
from dataset import make_dataset
from model import TasNet, TasNetParam

param = TasNetParam(N=500, L=40, H=1000, K=10, C=4, g=1.0, b=0.0)
model = TasNet.make(param)

epoch = 0
if path.exists("training"):
    checkpoints = [name for name in listdir("training") if "ckpt" in name]
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]
    epoch = int(checkpoint_name) + 1
    model.load_weights(f"training/{checkpoint_name}.ckpt")

checkpoint_path = "training/{epoch:05d}.ckpt"
while True:
    print("Epoch:", epoch)
    dataset = make_dataset(param, 5, 100, 1000)
    model.fit(dataset)
    model.save_weights(checkpoint_path.format(epoch=epoch))
    epoch += 1
