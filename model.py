import tensorflow as tf


class TasNetParam:
    __slots__ = 'N', 'L', 'H', 'K', 'C', 'g', 'b'

    def __init__(self, N: int, L: int, H: int, K: int, C: int, g: float, b: float):
        self.N, self.L, self.H, self.K, self.C = N, L, H, K, C
        self.g, self.b = g, b

    def get_config(self):
        return {
            "N": self.N,
            "L": self.L,
            "H": self.H,
            "K": self.K,
            "C": self.C,
            "g": self.g,
            "b": self.b
        }


class EncodingNet(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(EncodingNet, self).__init__(**kwargs)
        self.param = param
        self.input_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.K, self.param.L, 1))
        self.relu = tf.keras.layers.Conv2D(self.param.N,
                                           kernel_size=(1, self.param.L),
                                           activation="relu",
                                           padding="valid")
        self.sigmoid = tf.keras.layers.Conv2D(self.param.N,
                                              kernel_size=(1, self.param.L),
                                              activation="sigmoid",
                                              padding="valid")
        self.multiply = tf.keras.layers.Multiply()
        self.output_reshape = tf.keras.layers.Reshape(
            (self.param.K, self.param.N))
        self.normalization = tf.keras.layers.LayerNormalization()
        self.transform = tf.keras.layers.Lambda(
            lambda w: self.param.g * w + self.param.b)

    def call(self, inputs):
        outputs = self.input_reshape(inputs)
        relu = self.relu(outputs)
        sigmoid = self.sigmoid(outputs)
        outputs = self.multiply([relu, sigmoid])
        outputs = self.output_reshape(outputs)
        outputs = self.normalization(outputs)
        outputs = self.transform(outputs)
        return outputs

    def get_config(self):
        return self.param.get_config()


class SeparationNet(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(SeparationNet, self).__init__(**kwargs)
        self.param = param
        self.L1 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L2 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L3 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L4 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.skip_conn = tf.keras.layers.Add()
        self.M = tf.keras.layers.Dense(self.param.N * self.param.C)
        self.softmax = tf.keras.layers.Softmax(axis=-2)
        self.D = tf.keras.layers.Multiply()
        self.reshape = tf.keras.layers.Reshape((self.param.K, self.param.C, self.param.N))

    def call(self, inputs):
        L1_outputs = self.L1(inputs)
        L2_outputs = self.L2(L1_outputs)
        L3_outputs = self.L3(L2_outputs)
        L4_outputs = self.L4(L3_outputs)
        outputs = self.skip_conn([L2_outputs, L4_outputs])
        outputs = self.M(outputs)
        inputs = tf.keras.layers.concatenate([inputs for i in range(self.param.C)], axis=-1)
        outputs = self.D([inputs, outputs])
        outputs = self.reshape(outputs)
        return outputs

    def get_config(self):
        return self.param.get_config()

class DecodingNet(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(DecodingNet, self).__init__(**kwargs)
        self.param = param
        self.basis = tf.keras.layers.Dense(self.param.L)
        self.permute = tf.keras.layers.Permute((2, 1, 3))

    def call(self, inputs):
        outputs = self.basis(inputs)
        outputs = self.permute(outputs)
        return outputs

    def get_config(self):
        return self.param.get_config()


class SiSNR(tf.keras.losses.Loss):
    def __init__(self, param: TasNetParam, **kwargs):
        super(SiSNR, self).__init__(**kwargs)
        self.param = param

    def call(self, s, s_hat):
        s_hat = tf.reshape(s_hat, s.shape)
        s_target = s * (tf.reduce_sum(tf.multiply(s, s_hat)) / tf.norm(s))
        e_noise = s_hat - s_target
        result = 20 * tf.math.log(tf.norm(e_noise) /
                                  (tf.norm(s_target) + 1e-10) + 1e-10)
        return result


class TasNet(tf.keras.Model):
    @staticmethod
    def make(param: TasNetParam):
        model = TasNet(param)
        model.compile(optimizer="adam", loss=SiSNR(param))
        model.build(input_shape=(None, param.K, param.L))
        return model

    def __init__(self, param: TasNetParam, **kwargs):
        super(TasNet, self).__init__(**kwargs)
        self.param = param
        self.encoding = EncodingNet(self.param)
        self.separation = SeparationNet(self.param)
        self.decoding = DecodingNet(self.param)

    def call(self, inputs):
        outputs = self.encoding(inputs)
        separated = self.separation(outputs)
        decoded = self.decoding(separated)
        return decoded
