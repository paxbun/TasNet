import tensorflow as tf
from os import path, listdir
from config import get_param, get_directory_name
from dataset import make_dataset
from model import TasNet, TasNetParam, SDR

param = get_param()
directory_name = get_directory_name(param)
model = TasNet.make(param, tf.keras.optimizers.Adam(), SDR(param))

epoch = 0
if path.exists(directory_name):
    checkpoints = [name for name in listdir(
        directory_name) if "ckpt" in name]
    checkpoints.sort()
    checkpoint_name = checkpoints[-1].split(".")[0]
    epoch = int(checkpoint_name) + 1
    model.load_weights(f"{directory_name}/{checkpoint_name}.ckpt")

while True:
    checkpoint_path = f"{directory_name}/{epoch:05d}.ckpt"
    print(f"Epoch: {epoch}")
    dataset = make_dataset(param, 5, 100, 1000)
    history = model.fit(dataset)
    model.save_weights(checkpoint_path)
    epoch += 1
