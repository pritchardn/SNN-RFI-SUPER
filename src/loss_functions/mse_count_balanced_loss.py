"""
This module contains an implementation for an adjusts loss function present in snnTorch.
"""

import torch
from snntorch import spikegen
from snntorch.functional import LossFunctions
from torch import nn


class mse_count_loss_balanced(LossFunctions):
    """Balanced Mean Square Error Spike Count Loss.
    When called, the total spike count is accumulated over time for
    each neuron.
    The target spike count for correct classes is set to
    (num_steps * correct_rate), and for incorrect classes
    (num_steps * incorrect_rate).
    The spike counts and target spike counts are then applied to a
     Mean Square Error Loss Function.
    This function is adopted from SLAYER by Sumit Bam Shrestha and
    Garrick Orchard.
    This function is extended to handle un-classified targets (silence is golden).

    Example::

        import snntorch.functional as SF

        loss_fn = SF.mse_count_loss(correct_rate=0.75, incorrect_rate=0.25)
        loss = loss_fn(outputs, targets)


    :param correct_rate: Firing frequency of correct class as a ratio, e.g.,
        ``1`` promotes firing at every step; ``0.5`` promotes firing at 50% of
        steps, ``0`` discourages any firing, defaults to ``1``
    :type correct_rate: float, optional

    :param incorrect_rate: Firing frequency of incorrect class(es) as a
        ratio, e.g., ``1`` promotes firing at every step; ``0.5`` promotes
        firing at 50% of steps, ``0`` discourages any firing, defaults to ``1``
    :type incorrect_rate: float, optional

    :return: Loss
    :rtype: torch.Tensor (single element)

    """

    def __init__(
        self,
        correct_rate=1,
        incorrect_rate=0,
    ):
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.__name__ = "mse_count_loss_balanced"

    def __call__(self, spk_out, targets):
        _, num_steps, num_outputs = self._prediction_check(spk_out)
        loss_fn = nn.MSELoss()

        # generate ideal spike-count in C sized vector
        on_target = int(num_steps * self.correct_rate)
        off_target = int(num_steps * self.incorrect_rate)
        spike_count = torch.sum(spk_out, 0)  # B x C
        if targets.shape[0] == 0:
            spike_count_target = torch.ones_like(spike_count) * off_target
        else:
            spike_count_target = spikegen.targets_convert(
                targets,
                num_classes=num_outputs,
                on_target=on_target,
                off_target=off_target,
            )
        loss = loss_fn(spike_count, spike_count_target)
        return loss / num_steps
