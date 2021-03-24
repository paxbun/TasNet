import tensorflow as tf
import numpy as np
import musdb
import random
from model import TasNetParam

mus = list(musdb.DB(root="D:/Datasets/musdb18").tracks)

def musdb_generator(param: TasNetParam, batch_size: int, n: int):
    shape = (param.K, param.L)
    duration = param.K * param.L / 44100
    for _ in range(n):
        X = []
        Y = []
        for _ in range(batch_size):
            track = random.choice(mus)
            track.chunk_duration = duration
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            x = track.audio.T
            x_0, x_1 = np.reshape(x[0], shape), np.reshape(x[1], shape)
            y = [
                track.targets[target].audio.T
                for target in ("vocals", "drums", "bass", "other")
            ]
            y_0 = np.array([np.reshape(track[0], shape) for track in y])
            y_1 = np.array([np.reshape(track[1], shape) for track in y])
            X.extend([x_0, x_1])
            Y.extend([y_0, y_1])
        yield np.array(X), np.array(Y)

def make_dataset(param: TasNetParam, batch_size: int, n: int):
    return tf.data.Dataset.from_generator(lambda: musdb_generator(param, batch_size, n),
                                          output_signature=(
                                              tf.TensorSpec(shape=(batch_size * 2, param.K, param.L)),
                                              tf.TensorSpec(shape=(batch_size * 2, param.C, param.K, param.L))
                                          ))