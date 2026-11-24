"""RaCoFD: reliability-aware complementary Foundation/Detector distillation.

All RaCoFD components are training-only. RTv4's inference computation is unchanged.
"""

from .detection_teacher import RaCoFDDetectionTeacher
from .criterion import RaCoFDCriterion

__all__ = ['RaCoFDDetectionTeacher', 'RaCoFDCriterion']
