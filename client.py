import torch
from torch.multiprocessing import Process, Manager
from enum import Enum
import os
from manager import HybridPSManager
from typing import Dict
import datetime
import logging

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
  ON_FLIGHT = 1

  # 正在被使用，对tensor access之后的状态
  IN_USE = 2

  # 使用完毕，对tensor release之后的状态
  FREE = 3
############ CHUNK ###################

class TensorInfo(object):
  """
  记录chunk内存存储tensor的属性
  """
  def __init__(self, start : int, size : int, ps_id : int, status : PSTensorStatus):
    self.start = start
    self.size = size
    self.ps_id = ps_id
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

  def allocate(self, size : int, ps_id : int = None):
    """
    分配大小为size的连续存储空间，ps_id用于记录chunk存储tensor在Module中的位置
    """
    if self.capacity - self.offset < size:
      return None
    dest = self.payload.narrow(0, self.offset, size)
    self.tensor_infos.append(TensorInfo(self.offset, size, ps_id, PSTensorStatus.FREE))
    self.offset += size
    self.touch()
    return dest
  
  def visit(self):
    """
    展示Chunk内所有tensor信息
    """
    for info in self.tensor_infos:
      print(f"tensor in chunk start {info.start}, \
        end {info.start + info.size}, ps_id {info.ps_id}, status {info.status}")

  def move(self,
          param_dict : Dict[int, torch.nn.Parameter], 
          target_device : torch.device):
    """
    将这个chunk移动到device上，
    先要在target_device腾出空间
    """
    if self.device == target_device:
      return
    self.payload = self.payload.to(target_device)
    self.ps_manager.add(target_device.type, target_device.index, self.capacity)
    self.ps_manager.delete(self.device.type, self.device.index, self.capacity)
    # 将参数指针重新定位到新的设备上
    start = 0
    for info in self.tensor_infos:
      param = param_dict[info.ps_id]
      size = param.ps_numel
      param.ps_tensor =  self.payload.narrow(0, start, size).view(param.ps_shape)
      param.data = param.ps_tensor.data
      start += size
    self.device = target_device
    self.touch()

  def is_free():
    for info in self.tensor_infos:
      if info.status == PSTensorStatus.IN_USE:
        return False
      return True

######### HybridPS #############

class HybridPSClient(object):
  def __init__(self,
                gpu_index : int = 0, 
                data_type : torch.dtype = torch.float,
                default_chunk_size : int = 64):
    """
    管理一个Process的Param, AccGrad, OS数据。
    每个进程可以访问一个GPU的显存，和cpu的内存
    功能:
      1. 充分利用cpu和gpu内存
      2. 细粒度调度，HybridPSClient包含若干chunk
    """
    # index of gpu
    self.pid = os.getpid()
    self.gpu_index = gpu_index
    self.data_type = data_type

    self.chunk_list = []
    self.default_chunk_size = default_chunk_size

    self.module = None
    self.ps_id = 0
    self.params_dict = {}
    self.dict_param_id_chunk_id = {}

    self.cuda_buffered_chunk_size = 1
    self.cuda_buffered_chunk_list = []

  def prepare_device(self, target_device : torch.device, size : int):
    """
    TODO(jiaruifang)目前只考虑单GPU的情况
    """
    # param.compute_device上腾出足够的空间
    ps_manager = HybridPSManager()
    if ps_manager.max_mem(target_device.type, target_device.index) < size:
      logging.log(logging.ERROR, f"{target_device} has not enough space for {size} elements")
      raise RuntimeError
    while True:
      available_mem = ps_manager.available_mem(target_device.type, target_device.index)
      if available_mem >= size:
        break
      err = f"pid {self.pid}, available memory {available_mem} is less than chunk's capacity {size}"
      logging.log(logging.DEBUG, err)

      # TODO(jiaruifang) move out哪个chunk应该考虑时间戳
      for idx, chunk in enumerate(self.chunk_list):
        if chunk.device == target_device:
          self.chunk_move(idx, torch.device('cpu') if target_device.type == 'cuda' else torch.device('cuda:0'))
          break

  def access(self, param : torch.nn.Parameter):
    """
    访问一个module中的tensor，返回有正确数据的param
    找到param对应的chunk，然后决定是否移动chunk到本地设备
    移动之前要给设备腾出足够空间
    """
    if not self.is_ps_param(param):
      raise "access a param not ps_tensor through HybridPS API"

    chunk_id = self.dict_param_id_chunk_id[param.ps_id]
    if param.compute_device != param.ps_tensor.device:
      ps_id = param.ps_id
      self.prepare_device(param.compute_device, self.chunk_list[chunk_id].capacity)
      self.chunk_move(chunk_id, param.compute_device)
      assert param.compute_device == param.ps_tensor.device
    
    param.data = param.ps_tensor.data
    param.status = PSTensorStatus.IN_USE
    self.chunk_list[chunk_id].touch()

  def release(self, param : torch.nn.Parameter):
    """
    要不要把chunk迁移到cpu，如果它在gpu上的话，与access最后一句话冲突
    """
    param.status = PSTensorStatus.FREE

  def new_tensor(self, shape : torch.Size, ps_id : int = None):
    """
    在PS上新分配shape大小空间, ps_id表示param的id
    """
    numel = 1
    for elem in shape:
      numel *= elem
    
    chunk_id = 0
    if len(self.chunk_list) == 0:
      # 根据当前client所在设备为参考，使用manager调度获得一个最佳的device
      chunk_size = max(self.default_chunk_size, numel)
      self.chunk_list.append(Chunk(capacity = chunk_size))
      chunk_id = len(self.chunk_list) - 1
      
    dest = self.chunk_list[-1].allocate(numel, ps_id)
    if dest is None:
      chunk_size = max(self.default_chunk_size, numel)
      self.chunk_list.append(Chunk(capacity = chunk_size))
      dest = self.chunk_list[-1].allocate(numel, ps_id)
      chunk_id = len(self.chunk_list) - 1

    logging.log(logging.DEBUG, f'pid {self.pid}, allocates a tensor on chunk {chunk_id}')
    if ps_id is not None:
      self.dict_param_id_chunk_id[ps_id] = chunk_id
    return dest.view(shape), chunk_id

  @staticmethod
  def is_ps_param(parameter : torch.nn.Parameter):
    return hasattr(parameter, 'ps_id')
  
  def _convert_to_ps_param(self, param : torch.nn.Parameter):
    param.ps_id = self.ps_id # TODO(jiaruifang) generate a id
    param.ps_numel = param.numel()
    param.ps_shape = param.shape
    param.ps_tensor = None

    # param所在的计算设备，计算现在指FWD，BWD，step
    param.compute_device = param.device

    # 如果ps_tensor已经存在了，则将param删除
    if param.ps_tensor is not None:
      param.data = torch.ones(1).half().to(param.compute_device)

    # 初始化ps_tensor空间
    if param.ps_tensor is None:
      param.ps_tensor, param.ps_chunk_id = self.new_tensor(param.shape, self.ps_id)

    # 拷贝param数据到ds_tensor上
    one_dim_param = param.contiguous().view(-1)
    param.ps_tensor.copy_(one_dim_param.view(param.ps_shape))
    
    # 将原来数据删除，并指向payload空间
    # param.data = torch.ones(1).half().to(param.compute_device)
    param.data = param.ps_tensor

    # 注册到Client类中
    self.params_dict[self.ps_id] = param
    self.ps_id += 1

  def register_module(self, module : torch.nn.Module):
    """
    将模型每个layer的param由HybridPS管理
    """
    if module is not None:
      assert isinstance(module, torch.nn.Module)
      self.module = module
      for param in module.parameters(recurse=True):
          if self.is_ps_param(param):
            continue
          self._convert_to_ps_param(param)

  def register_tensor(self, src_tensor : torch.Tensor):
    """
    @deprecated, used for debug
    Register a tensor to HybridPSClient's payload.
    Tensors are flatten and concated in a contigous memory space.
    """
    if self.is_ps_param(src_tensor):
      return
    self._convert_to_ps_param(src_tensor)

  def visit(self):
    for idx, chunk in enumerate(self.chunk_list):
      print(f"chunk {idx} on device {chunk.device}")
      chunk.visit()

  def chunk_move(self, chunk_id : int, device : torch.device):
    """
    测试函数，将chunk_id的chunk移动到gpu上
    需要对对应param重新赋值
    """
    if self.chunk_list[chunk_id].device != device:
      logging.log(logging.DEBUG, f'pid {self.pid} move chunk {chunk_id} from {self.chunk_list[chunk_id].device} to {device}')
      self.chunk_list[chunk_id].move(self.params_dict, device)

  def allreduce(self, local_tensor):
    """
    必须所有process同时执行，规约后的payload存储在哪(cpu or gpu)由调度器决定
    """
    pass

  def broadcast(self, local_tensor : torch.Tensor):
    """
    必须所有process同时执行，规约后的payload存储在哪由调度器决定
    """
    pass

