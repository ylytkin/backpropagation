"""
neural_network.py

An implementation of a simple feed-forward neural network
with sigmoid activations and Adam optimizer.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

__all__ = [
    'NeuralNetwork',
]


class NeuralNetwork:
    """An implementation of a simple feed-forward neural network
    with sigmoid activations and Adam optimizer.
    """

    def __init__(self, layer_sizes: np.ndarray):
        """Constructor.

        `layer_sizes` is an array of layer sizes, including
        the input layer as the first element.

        :param layer_sizes: layer sizes
        """

        self.layer_sizes = layer_sizes
        self.layer_ids = np.arange(1, layer_sizes.size)
        self.input_layer_id = 0
        self.output_layer_id = self.layer_ids[-1]

        self.weights = None
        self.biases = None

        self.alpha = None
        self.beta1 = None
        self.beta2 = None
        self.epsilon = None

        self.weight_first_moments = None
        self.weight_second_moments = None

        self.bias_first_moments = None
        self.bias_second_moments = None

        self.compiled = False

    def compile(
            self,
            alpha: float = 0.001,
            beta1: float = 0.9,
            beta2: float = 0.999,
            epsilon: float = 10 ** (-8),
    ) -> 'NeuralNetwork':
        """Compile the model with random weights and zero biases.

        Arguments `alpha`, `beta1`, `beta2`, and `epsilon` are the
        hyperparameters of the Adam optimizer. People tend to just
        leave them as they are.
        """

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        sigma = np.sqrt(2 / self.layer_sizes[-2])

        self.weights = dict()
        self.biases = dict()

        self.weight_first_moments = dict()
        self.weight_second_moments = dict()

        self.bias_first_moments = dict()
        self.bias_second_moments = dict()

        for i in self.layer_ids:
            weights_shape = (self.layer_sizes[i], self.layer_sizes[i - 1])
            biases_shape = (self.layer_sizes[i], 1)

            self.weights[i] = np.random.normal(loc=0, scale=sigma, size=weights_shape)
            self.weight_first_moments[i] = np.zeros(weights_shape)
            self.weight_second_moments[i] = np.zeros(weights_shape)

            self.biases[i] = np.zeros(biases_shape)
            self.bias_first_moments[i] = np.zeros(biases_shape)
            self.bias_second_moments[i] = np.zeros(biases_shape)

        self.compiled = True

        return self

    def get_layer_outputs(self, x: np.ndarray) -> Dict[int, np.ndarray]:
        """Get outputs of all layers. The input is also included
        for convenience.

        The array `x` is assumed to be consisting of objects as columns
        (and, consequently, features of rows).

        :param x: input
        :return: dict of layer outputs
        """

        if len(x.shape) == 1:
            a = x.reshape(-1, 1)
        else:
            a = x.copy()

        layer_outputs = {self.input_layer_id: a}

        for layer_id in self.layer_ids:
            a = self.sigmoid(self.weights[layer_id].dot(a) + self.biases[layer_id])
            layer_outputs[layer_id] = a

        return layer_outputs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return model predictions on input `x`.

        The array `x` is assumed to be consisting of objects as rows
        (and, consequently, features as columns).

        :param x: input
        :return: array of predictions
        """

        return self.get_layer_outputs(x.T)[self.output_layer_id].flatten()

    def apply_gradient_descent_step_on_layer(
            self,
            layer_id: int,
            weights_gradient: np.ndarray,
            biases_gradient: np.ndarray,
    ) -> None:
        """Apply gradient descent step on the `layer_id`-th layer.

        This is done using the Adam optimization method:
            https://arxiv.org/abs/1412.6980

        :param layer_id: layer id
        :param weights_gradient: weights gradient vector
        :param biases_gradient: biases gradient vector
        :return: None
        """

        self.weight_first_moments[layer_id] = (
                self.beta1 * self.weight_first_moments[layer_id]
                + (1 - self.beta1) * weights_gradient
        )
        self.weight_second_moments[layer_id] = (
                self.beta2 * self.weight_second_moments[layer_id]
                + (1 - self.beta2) * weights_gradient ** 2
        )

        self.bias_first_moments[layer_id] = (
                self.beta1 * self.bias_first_moments[layer_id]
                + (1 - self.beta1) * biases_gradient
        )
        self.bias_second_moments[layer_id] = (
                self.beta2 * self.bias_second_moments[layer_id]
                + (1 - self.beta2) * biases_gradient ** 2
        )
        
        # warning: apparently these are biased estimates for moments
        weights_step = (self.weight_first_moments[layer_id]
                        / (self.weight_second_moments[layer_id] ** 0.5 + self.epsilon))
        biases_step = (self.bias_first_moments[layer_id]
                       / (self.bias_second_moments[layer_id] ** 0.5 + self.epsilon))

        self.weights[layer_id] -= self.alpha * weights_step
        self.biases[layer_id] -= self.alpha * biases_step

    def fit_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int) -> None:
        """Fit a batch of input data.

        The array `x` is assumed to be consisting of objects as columns
        (and, consequently, features of rows).

        :param x: input data
        :param y: input labels
        :param batch_size: batch size
        :return: None
        """

        layer_outputs = self.get_layer_outputs(x)
        derivative = None

        for layer_id in self.layer_ids[::-1]:
            layer_output = layer_outputs[layer_id]
            previous_layer_output = layer_outputs[layer_id - 1]

            if layer_id == self.output_layer_id:
                derivative = layer_output - y.reshape(1, -1)
            else:
                derivative = self.weights[layer_id + 1].T.dot(derivative) \
                             * layer_output * (1 - layer_output)

            weights_gradient = derivative.dot(previous_layer_output.T) / batch_size
            biases_gradient = derivative.mean(axis=1, keepdims=True)

            self.apply_gradient_descent_step_on_layer(
                layer_id=layer_id,
                weights_gradient=weights_gradient,
                biases_gradient=biases_gradient,
            )

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int,
            batch_size: int = 1,
            verbose: bool = True,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, List[float]]:
        """Fit.

        :param x_train: train data
        :param y_train: train labels
        :param epochs: number of epochs
        :param batch_size: batch size
        :param verbose: whether to do verbose logging
        :param validation_data: optional validation data
        :return: training history (dict)
        """

        history = {
            'loss': [],
            'accuracy': [],
        }

        if validation_data is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        for epoch in range(1, epochs + 1):
            log = f'Epoch {self.justify_epoch_string(epoch, epochs=epochs)}: '

            for x, y in self.split_into_batches(x_train.T, y_train,
                                                batch_size=batch_size, shuffle=True):
                self.fit_batch(x, y, batch_size=batch_size)

            y_pred = self.predict(x_train)

            loss = self.loss(y_train, y_pred)
            accuracy = self.accuracy(y_train, y_pred > 0.5)

            history['loss'].append(loss)
            history['accuracy'].append(accuracy)

            log += (f'loss {self.justify_score_string(loss)}, '
                    f'acc {self.justify_score_string(accuracy)}')

            if validation_data is not None:
                x_val, y_val = validation_data

                y_pred = self.predict(x_val)

                val_loss = self.loss(y_val, y_pred)
                val_accuracy = self.accuracy(y_val, y_pred > 0.5)

                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                log += (f', val loss {self.justify_score_string(loss)}, '
                        f'val acc {self.justify_score_string(accuracy)}')

            if verbose is True:
                print(log)

        return history

    def split_into_batches(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int,
            shuffle: bool = True,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split the data into batches.

        The array `x` is assumed to be consisting of objects as columns
        (and, consequently, features of rows).

        :param x: input data
        :param y: input labels
        :param batch_size: batch size
        :param shuffle: whether to shuffle data
        :return: list of tuples of batch data and labels
        """

        if shuffle is True:
            x, y = self.shuffle(x, y)

        ids = np.arange(x.shape[1])

        batch_ids = [
            ids[offset : offset + batch_size]
            for offset in np.arange(0, ids.shape[0], batch_size)
        ]

        batches = [
            (x[:, current_batch_ids], y[current_batch_ids])
            for current_batch_ids in batch_ids
        ]

        return batches

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """The sigmoid function.
        """

        return np.exp(-np.logaddexp(0, -x))

    @staticmethod
    def loss(y_real: np.ndarray, probas: np.ndarray) -> float:
        """The Log Loss function.
        """

        return (- y_real * np.log(probas) - (1 - y_real) * np.log(1 - probas)).mean()

    @staticmethod
    def accuracy(y_real: np.ndarray, y_pred: np.ndarray) -> float:
        """Accuracy of the given predictions.
        """

        return (y_real == y_pred).sum() / y_real.size

    @staticmethod
    def shuffle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Shuffle the data.

        The array `x` is assumed to be consisting of objects as columns
        (and, consequently, features of rows).

        :param x: input data
        :param y: input labels
        :return: shuffled data and labels
        """

        permutation = np.random.permutation(x.shape[1])

        return x[:, permutation], y[permutation]

    @staticmethod
    def justify_epoch_string(epoch: int, epochs: int) -> str:
        """Justify the epoch string for logging.

        :param epoch: current epoch
        :param epochs: total number of epochs
        :return: epoch string
        """

        n = len(str(epochs))

        return str(epoch).ljust(n, ' ')

    @staticmethod
    def justify_score_string(value: float) -> str:
        """Justify the score string for logging.

        :param value: score
        :return: score string
        """

        return str(round(value, 4)).ljust(6, '0')
