import pandas as pd
import os
from pathlib import Path
import json
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    def __init__(self, path):
        super(BaseLogger, self).__init__()
        self.path = path