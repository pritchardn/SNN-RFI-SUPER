"""
Fully connected ANN model for spectrograms. Operated in patch mode.
"""

from models.fc_ann import LitFcANN


class LitFcPANN(LitFcANN):
    def __init__(
        self, num_inputs: int, num_hidden: int, num_outputs: int, num_layers: int
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, num_layers)
