
from enum import Enum

class AccessType(Enum):
  DATA = 1
  GRAD = 2

class PSChunkStatus(Enum):
  # Chunk只在cpu上
  CPU_ONLY = 1

  # Chunk只在gpu上
  GPU_ONLY = 2

  # Chunk复制两份，分别在cpu和gpu上
  CPU_GPU = 3

  # chunk的本体是在远程GPU上, release全部删除
  FROM_REMOTE_GPU = 4

class PSTensorStatus(Enum):
  # 在拷贝中
  DATA_ON_FLY = 1
  GRAD_ON_FLY = 2

  DATA_IN_USE = 3
  GRAD_IN_USE = 4
  
  DATA_HOLD = 5
  GRAD_HOLD = 6
  FREE = 7