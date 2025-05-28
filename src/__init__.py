"""
FedVideoQA: Federated Learning Framework for Video Question Answering

A privacy-preserving multimodal learning system that enables collaborative
VideoQA development across institutions while maintaining strict data protection.
"""

__version__ = "1.0.0"
__author__ = "FedVideoQA Team"
__email__ = "contact@fedvideoqa.org"

from .core import *
from .models import *
from .federated import *
from .privacy import *
from .utils import * 