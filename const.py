
from enum import Enum

class AccessType(Enum):
  DATA = 1
  GRAD = 2

class PSChunkStatus(Enum):
  # Chunk只在cpu上
  COMPUTE = 1
  HOLD = 2
  FREE = 3

class PSTensorStatus(Enum):
  # 正在被用于计算，不能随意迁移
  COMPUTE = 1
  # 可以迁移，不能释放
  HOLD = 2
  # 可以释放
  FREE = 3