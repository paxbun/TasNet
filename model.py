import tensorflow as tf


class TasNetParam:
    __slots__ = 'N', 'L', 'H', 'K', 'C', 'g', 'b'

    def __init__(self, N: int, L: int, H: int, K: int, C: int, g: float, b: float):
        self.N, self.L, self.H, self.K, self.C = N, L, H, K, C
        self.g, self.b = g, b


class EncodingNet(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(EncodingNet, self).__init__(**kwargs)
        self.param = param
        self.input_reshape = tf.keras.layers.Reshape(target_shape=(self.param.K, self.param.L, 1))
        self.relu = tf.keras.layers.Conv2D(self.param.N,
                                           kernel_size=(1, self.param.L),
                                           activation="relu",
                                           padding="valid")
        self.sigmoid = tf.keras.layers.Conv2D(self.param.N,
                                              kernel_size=(1, self.param.L),
                                              activation="relu",
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
        sigmoid = self.relu(outputs)
        outputs = self.multiply([relu, sigmoid])
        outputs = self.output_reshape(outputs)
        outputs = self.normalization(outputs)
        outputs = self.transform(outputs)
        return outputs


class SeparationNetLSTM(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(SeparationNetLSTM, self).__init__(**kwargs)
        self.param = param
        self.L1 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L2 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L3 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.L4 = tf.keras.layers.LSTM(self.param.H, return_sequences=True)
        self.SkipConn = tf.keras.layers.Add()

    def call(self, inputs):
        L1_outputs = self.L1(inputs)
        L2_outputs = self.L2(L1_outputs)
        L3_outputs = self.L3(L2_outputs)
        L4_outputs = self.L4(L3_outputs)
        outputs = self.SkipConn([L2_outputs, L4_outputs])
        return outputs


class SeparationNetDense(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(SeparationNetDense, self).__init__(**kwargs)
        self.param = param
        self.M = tf.keras.layers.Dense(self.param.N)
        self.D = tf.keras.layers.Multiply()

    def call(self, inputs):
        W, LSTM = inputs
        outputs = self.M(LSTM)
        outputs = self.D([W, outputs])
        return outputs


class DecodingNet(tf.keras.layers.Layer):
    def __init__(self, param: TasNetParam, **kwargs):
        super(DecodingNet, self).__init__(**kwargs)
        self.param = param
        self.basis = tf.keras.layers.Dense(self.param.L)

    def call(self, inputs):
        outputs = self.basis(inputs)
        return outputs


class SiSNR(tf.keras.losses.Loss):
    def __init__(self, param: TasNetParam, **kwargs):
        super(SiSNR, self).__init__(**kwargs)
        self.param = param

    def call(self, s, s_hat):
        s_hat = tf.reshape(s_hat, s.shape)
        s_target = s * (tf.reduce_sum(tf.multiply(s, s_hat)) / tf.norm(s))
        e_noise = s_hat - s_target
        result = 20 * tf.math.log(tf.norm(e_noise) / (tf.norm(s_target) + 1e-10) + 1e-10)
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
        self.separation_lstm = SeparationNetLSTM(self.param)
        self.separation_dense_list = [
            SeparationNetDense(self.param)
            for i in range(self.param.C)
        ]
        self.decoding = DecodingNet(self.param)

    def call(self, inputs):
        outputs = self.encoding(inputs)
        lstm_outputs = self.separation_lstm(outputs)
        separated_list = [
            separation_dense([outputs, lstm_outputs])
            for separation_dense in self.separation_dense_list
        ]
        decoded_list = [
            self.decoding(separated)
            for separated in separated_list
        ]
        return tf.keras.layers.concatenate(decoded_list, axis=-3)
