from typing import List

import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

from root.src.domain.model_config import ModelConfig

class NeuralNetwork:
    "Адаптер domain-конфигурации к конкретной реализации на TensorFlow"

    def __init__(self, model_config: ModelConfig) -> tf.keras.Model:
        self.__model_config: ModelConfig = model_config
        self.__model: tf.keras.Model = self.__build_model()

    @property
    def model(self) -> tf.keras.Model:
        return self.__model

    def __build_model(self):
        model_layers: List[tf.keras.layers.Layer] = []

        model_layers.append(
            tf.keras.layers.Input(shape=(self.__model_config.input_size,))
        )

        for units in self.__model_config.hidden_layers:
            model_layers.append(
                tf.keras.layers.Dense(
                    units=units, 
                    activation=self.__model_config.activation,
                    use_bias=self.__model_config.use_bias,
                    kernel_initializer=self.__he_initializer()
                )
            )

            # Деактивирующий слой
            if self.__model_config.dropout_rate > 0:
                model_layers.append(tf.keras.layers.Dropout(self.__model_config.dropout_rate))

        model_layers.append(
            tf.keras.layers.Dense(
                units=self.__model_config.output_size,
                use_bias=self.__model_config.use_bias
            )
        )

        return tf.keras.Sequential(model_layers, name="feedforward_network")

    def visualize_weights_distribution(self) -> None:
        weights = []

        for layer in self.__model.layers:
            if hasattr(layer, "kernel"):
                weights.append(layer.kernel.numpy().flatten())
        
        if not weights:
            print("no trainable kernels founds")
            return
        
        plt.figure()
        plt.hist(np.concatenate(weights), bins=50)
        plt.title("Таблица распределения весов")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.show()

    @staticmethod
    def __he_initializer() -> tf.keras.initializers.Initializer:
        return tf.keras.initializers.HeNormal()