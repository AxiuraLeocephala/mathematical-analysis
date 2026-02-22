from dataclasses import dataclass
from typing import Tuple, Literal

Activation = Literal["relu", "sigmoid", "tanh", "gelu"]
OptimizerType = Literal["sgd", "adam", "rmsprop"]
DeviceType = Literal["cpu", "gpu", "auto"]

@dataclass(frozen=True)
class ModelConfig:
    "Конфигурация архитектуры сети"
    
    input_size: int # Количество входов
    hidden_layers: Tuple[int, ...] # Количество слоев
    output_size: int # Количество выходов

    activation: Activation # Функция активации узла сети
    use_bias: bool # Использовать смещение?
    dropout_rate: float # Процент отсева

@dataclass(frozen=True)
class TrainingConfig:
    "Конфигурация процесса обучения"

    learning_rate: float # Скорость обучения
    batch_size: int # Размер партии
    epochs: int # Количество эпох
    
    optimizer: OptimizerType # Оптимизатор
    weight_decay: float # Значение изменения веса
    shuffle: bool # Использовать тасовку обучающих данных?

@dataclass(frozen=True)
class ExperimentConfig:
    "Конфигурация эксперимента"

    model: ModelConfig
    training: TrainingConfig

    seed: int
    device: DeviceType
    experiment_name: str

def validate_model_config(config: ModelConfig) -> None:
    "Доменный валидатор модели"

    if config.input_size <= 0:
        raise ValueError("input_size must be > 0")

    if config.output_size <= 0:
        raise ValueError("output_size must be > 0")

    if not config.hidden_layers:
        raise ValueError("hidden_layers must not be empty")

    if any(size <= 0 for size in config.hidden_layers):
        raise ValueError("hidden layer sizes must be > 0")

    if not (0.0 <= config.dropout_rate < 1.0):
        raise ValueError("dropout_rate must be in [0, 1)")