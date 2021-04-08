import torch
from torch.multiprocessing import Process, Manager
from enum import Enum
import os
from manager import HybridPSManager
from typing import Dict
import datetime
import logging

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
############ CHUNK ###################

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

  def allocate(self, size : int, tensor_id : int = None):
    """
    分配大小为size的连续存储空间，tensor_id用于记录chunk存储tensor在Module中的位置
    """
    if self.capacity - self.offset < size:
      return None
    dest = self.payload.narrow(0, self.offset, size)
    self.tensor_infos.append(TensorInfo(self.offset, size, tensor_id, PSTensorStatus.FREE))
    self.offset += size
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
    start = 0
    for info in self.tensor_infos:
      tensor_id = info.tensor_id
      if tensor_id in param_data_dict.keys():
        logging.debug(f'chunk moves data tensor {tensor_id} to {target_device}')
        param = param_data_dict[tensor_id]
        size = param.ps_numel
        param.ps_data_tensor =  self.payload.narrow(0, start, size).view(param.ps_shape)
        param.data = param.ps_data_tensor
      elif tensor_id in param_grad_dict.keys():
        logging.debug(f'chunk moves grad tensor {tensor_id}')
        param = param_grad_dict[tensor_id]
        size = param.ps_numel
        param.ps_grad_tensor =  self.payload.narrow(0, start, size).view(param.ps_shape)
        param.grad = param.ps_grad_tensor
      start += size
    self.device = target_device
    self.touch()

  def is_free():
    for info in self.tensor_infos:
      if info.status != PSTensorStatus.FREE:
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
    self.ps_id = -1
    self.param_data_dict = {}
    self.param_grad_dict = {}
    self.dict_tensor_id_chunk_id = {}

    self.cuda_buffered_chunk_size = 1
    self.cuda_buffered_chunk_list = []

  def prepare_device(self, target_device : torch.device, size : int):
    """
    TODO(jiaruifang)目前只考虑单GPU的情况
    """
    logging.debug(f'prepare_device {target_device}')
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

  def access(self, param : torch.nn.Parameter, access_type : AccessType):
    """
    访问一个module中的tensor，返回有正确数据的param
    找到param对应的chunk，然后决定是否移动chunk到本地设备
    移动之前要给设备腾出足够空间
    """
    if not self.is_ps_param(param):
      raise "access a param not ps_data_tensor through HybridPS API"
    
    logging.debug('access')
    # tensor_id to chunk_id
    if access_type == AccessType.DATA:
      chunk_id = self.dict_tensor_id_chunk_id[param.ps_data_id]
      current_device = param.data.device
      logging.debug(f'access, tensor id {param.ps_data_id}, chunk id {chunk_id} current device {current_device}, while target {param.compute_device}')
    elif access_type == AccessType.GRAD:
      chunk_id = self.dict_tensor_id_chunk_id[param.ps_grad_id]
      current_device = param.grad.device
    else:
      raise RuntimeError

    if param.compute_device != current_device:
      logging.debug('begin chunk move')
      self.prepare_device(param.compute_device, self.chunk_list[chunk_id].capacity)
      self.chunk_move(chunk_id, param.compute_device)

    self.visit()

    if access_type == AccessType.DATA:
      current_device = param.data.device
    elif access_type == AccessType.GRAD:
      current_device = param.grad.device
    else:
      raise RuntimeError

    logging.debug(f'current dev {current_device}, compute dev {param.compute_device}')
    assert current_device == param.compute_device

    self.chunk_list[chunk_id].touch()

    if access_type == AccessType.DATA:
      param.data = param.ps_data_tensor.data
      param.status = PSTensorStatus.DATA_IN_USE
    elif access_type == AccessType.GRAD:
      param.grad = param.ps_grad_tensor.data
      param.status = PSTensorStatus.GRAD_IN_USE

  def access_data(self, param : torch.nn.Parameter):
    self.access(param, AccessType.DATA)

  def access_grad(self, param : torch.nn.Parameter):
    self.access(param, AccessType.GRAD)

  def release(self, param : torch.nn.Parameter, access_type : AccessType):
    """
    这个param的data, grad不再需要放在计算设备，或者不需要hold
    TODO(jiaruifang)释放内存 or 只是不再计算设备的hold
    """
    if access_type == AccessType.DATA:
      param.status = PSTensorStatus.DATA_HOLD
    elif access_type == AccessType.GRAD:
      param.grad = param.ps_grad_tensor.data
      param.status = PSTensorStatus.GRAD_HOLD

  def release_data(self, param : torch.nn.Parameter):
    self.release(param, AccessType.DATA)

  def release_grad(self, param : torch.nn.Parameter):
    self.release(param, AccessType.GRAD)

  def new_tensor(self, shape : torch.Size, tensor_id : int):
    """
    在PS上新分配shape大小空间, tensor_id是tensor在本进程内唯一标识
    TODO(jiaruifang) 现在的分配方式很简单，没考虑chunk空间可以释放的情况。
    只检查最后一个chunk是否有空余，如果没有分配新的chunk
    这个函数最后要注册tensor_id和chunk_id的对应关系，
    未来需要用tensor_id来索引chunk_id，chunk_id索引chunk
    chunk_list顺序递增
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
      
    dest = self.chunk_list[-1].allocate(numel, tensor_id)
    chunk_id = len(self.chunk_list) - 1
    if dest is None:
      chunk_size = max(self.default_chunk_size, numel)
      self.chunk_list.append(Chunk(capacity = chunk_size))
      dest = self.chunk_list[-1].allocate(numel, tensor_id)
      chunk_id = len(self.chunk_list) - 1

    logging.log(logging.DEBUG, f'pid {self.pid}, allocates a tensor on chunk {chunk_id}')
    if tensor_id is not None:
      self.dict_tensor_id_chunk_id[tensor_id] = chunk_id
    return dest.view(shape), chunk_id

  @staticmethod
  def is_ps_param(parameter : torch.nn.Parameter):
    return hasattr(parameter, 'ps_data_id')
  
  def generate_id(self):
    self.ps_id = self.ps_id + 1
    return self.ps_id 

  def _convert_to_ps_param(self, param : torch.nn.Parameter):
    """
    为param的data和grad分配空间
    """
    param.ps_numel = param.numel()
    param.ps_shape = param.shape
    
    param.ps_data_id = self.generate_id()
    param.ps_data_tensor = None

    param.ps_grad_id = self.generate_id()
    param.ps_grad_tesnor = None

    # param所在的计算设备，计算现在指FWD，BWD，step
    param.compute_device = param.device

    # 初始化ps_data_tensor空间，并向其拷贝数据
    param.ps_data_tensor, param.ps_data_chunk_id = self.new_tensor(param.shape, param.ps_data_id)
    one_dim_param = param.data.contiguous().view(-1)
    param.ps_data_tensor.copy_(one_dim_param.view(param.ps_shape))
    param.data = param.ps_data_tensor.data

    # 初始化ps_grad_tensor空间，并向其拷贝数据
    param.ps_grad_tensor, param.ps_gard_chunk_id = self.new_tensor(param.shape, param.ps_grad_id)
    if param.grad is not None:
      one_dim_grad = param.grad.contiguous().view(-1)
      param.ps_grad_tesnor.copy_(one_dim_grad.view(param.ps_shape))
      param.grad = param.ps_grad_tesnor.data

    # 注册到Client类中, tensor id -> param
    self.param_data_dict[param.ps_data_id] = param
    self.param_grad_dict[param.ps_grad_id] = param

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

  def register_param(self, src_param : torch.nn.Parameter):
    """
    @deprecated, used for debug
    Register a parameter to HybridPSClient's payload.
    Tensors (data, grad) in Param are flatten and concated in a contigous memory space.
    """
    if self.is_ps_param(src_param):
      return
    self._convert_to_ps_param(src_param)

  def visit(self):
    for idx, chunk in enumerate(self.chunk_list):
      print(f"chunk {idx} on device {chunk.device}")
      chunk.visit()

  def chunk_move(self, chunk_id : int, device : torch.device):
    """
    将chunk_id的chunk移动到device上
    需要对对应param重新赋值
    """
    logging.debug(f'chunk_move chunk id {chunk_id} from {self.chunk_list[chunk_id].device} to {device}')
    if self.chunk_list[chunk_id].device != device:
      logging.log(logging.DEBUG, f'pid {self.pid} move chunk {chunk_id} from {self.chunk_list[chunk_id].device} to {device}')
      self.chunk_list[chunk_id].move(self.param_data_dict, 
                                     self.param_grad_dict, 
                                     device)

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

