import os
import torch
from const import PSTensorStatus
from typing import Dict
from manager import HybridPSManager
import datetime
import logging

class TensorInfo(object):
  """
  记录chunk内存存储tensor的属性
  """
  def __init__(self, start : int, size : int, tensor_id : int, status : PSTensorStatus):
    self.start = start
    self.size = size
    self.tensor_id = tensor_id
    self.status = status
    
class Chunk(object):
  def __init__(self, capacity : int = 100):
    """
    Chunk是数据迁移的最小单位，
    它用一段连续的内存来存储张量
    Chunk可以有三种状态，只在CPU，只在GPU，CPU和GPU各有一份副本
    TODO(jiaruifang) 释放tensor带来的碎片问题，尚未实现存储空间的释放
    TODO(jiaruifang) capacity单位不是Byte而是可以存储float的个数
    """
    self.pid = os.getpid()

    self.capacity = capacity
    self.offset = 0

    # 由调度决定分配在CPU还是GPU上
    self.ps_manager = HybridPSManager()
    if self.ps_manager.is_init() is False:
      raise "init Manager first before init a Chunk"
    
    self.cuda_idx = torch.cuda.current_device()

    self.device = self.ps_manager.schedule(capacity, self.cuda_idx)
    if self.device.type == 'cuda' and self.device.index > torch.cuda.device_count():
      logging.log(logging.WARNING, "When init a Chunk, the assigned cuda device idx is larger than available cuda count")
      logging.log(logging.WARNING, "Set cuda index to 0")
      self.device = torch.cuda.current_device()

    self.payload = torch.zeros(capacity, dtype = torch.float, device = self.device)
    self.ps_manager.add(self.device.type, self.device.index, capacity)

    self.tensor_infos = []
    self.timestamp = datetime.datetime.now().timestamp()

  def touch(self):
    self.timestamp = datetime.datetime.now().timestamp()

  def get_timestamp(self):
    return self.timestamp

  def try_allocate(self, size : int, tensor_id : int) -> torch.Tensor:
    """
    在chunk的连续payload中找到一个满足size大小的碎片
    采用贪心算法，因为考虑NN参数的分配一般是连续分配，连续释放，没必要设计更复杂的算法
    """
    prev_end = 0
    for info in self.tensor_infos:
      start = info.start
      gap = start - prev_end
      if gap >= size:
        dest = self.allocate(start, size, tensor_id)
        return dest
      prev_end = start + info.size

    if self.capacity - prev_end >= size:
      dest = self.allocate(prev_end, size, tensor_id)
      return dest
    return None

  def allocate(self, offset : int, size : int, tensor_id : int = None):
    """
    分配大小为size的连续存储空间，tensor_id用于记录chunk存储tensor在Module中的位置
    """
    dest = self.payload.narrow(0, self.offset, size)
    self.tensor_infos.append(TensorInfo(self.offset, size, tensor_id, PSTensorStatus.FREE))
    self.touch()
    return dest
  
  def visit(self):
    """
    展示Chunk内所有tensor信息
    """
    for info in self.tensor_infos:
      print(f"tensor in chunk start {info.start}, \
        end {info.start + info.size}, tensor_id {info.tensor_id}, status {info.status}")

  def move(self,
          param_data_dict : Dict[int, torch.nn.Parameter], 
          param_grad_dict : Dict[int, torch.nn.Parameter], 
          target_device : torch.device):
    """
    将这个chunk移动到device上，
    先要在target_device腾出空间
    """
    logging.debug(f'move this chunk to {target_device}')
    if self.device == target_device:
      return
    self.payload = self.payload.to(target_device)
    self.ps_manager.add(target_device.type, target_device.index, self.capacity)
    self.ps_manager.delete(self.device.type, self.device.index, self.capacity)
    # 将参数指针重新定位到新的设备上
    for info in self.tensor_infos:
      tensor_id = info.tensor_id
      start = info.start
      size = info.size
      if tensor_id in param_data_dict.keys():
        logging.debug(f'chunk moves data tensor {tensor_id} to {target_device}')
        param = param_data_dict[tensor_id]
        param.ps_data_tensor =  self.payload.narrow(0, start, size).view(param.ps_shape)
        param.data = param.ps_data_tensor
      elif tensor_id in param_grad_dict.keys():
        logging.debug(f'chunk moves grad tensor {tensor_id} to {target_device}')
        param = param_grad_dict[tensor_id]
        param.ps_grad_tensor =  self.payload.narrow(0, start, size).view(param.ps_shape)
        param.grad = param.ps_grad_tensor

    self.device = target_device
    self.touch()

  def is_free():
    for info in self.tensor_infos:
      if info.status != PSTensorStatus.FREE:
        return False
      return True
