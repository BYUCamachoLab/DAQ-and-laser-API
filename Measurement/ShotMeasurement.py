
from Measurement.Measurement import Measurement
from abc import ABC, abstractmethod
import os
import random
from matplotlib import pyplot as plt
import numpy as np


class ShotMeasurement(Measurement, ABC):

    def __init__(self):
        super().__init__()


    def _perform_measurement(self):
        pass

    def _visualize_data(self, save_figure):
        pass

    def _save_data(self):
        pass

    def _get_data(self):
        pass
