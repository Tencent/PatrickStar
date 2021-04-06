import torch
from torch.multiprocessing import Process, Manager

from manager import HybridPSManager
from typing import Dict

############ CHUNK ###################

class TensorInfo(object):
  """
  记录chunk内存存储tensor的属性
  """
  def __init__(self, start : int, size : int, ps_id : int):
    self.start = start
    self.size = size
    self.ps_id = ps_id
    
class Chunk(object):
  def __init__(self, capacity : int = 100, device_type : torch.device = torch.device('cpu')):
    """
    Chunk是数据迁移的最小单位，
    它用一段连续的内存来存储张量
    TODO(jiaruifang) 释放tensor带来的碎片问题，尚未实现存储空间的释放
    TODO(jiaruifang) capacity单位不是Byte而是可以存储float的个数
    """
    self.capacity = capacity
    self.offset = 0
    self.device_type : torch.device = device_type
    self.payload = torch.zeros(capacity, dtype = torch.float, device = device_type)
    self.tensor_infos = []
      
  def allocate(self, size : int, ps_id : int = None):
    """
    分配大小为size的连续存储空间，ps_id用于记录chunk存储tensor在Module中的位置
    """
    if self.capacity - self.offset < size:
      return None
    dest = self.payload.narrow(0, self.offset, size)
    self.tensor_infos.append(TensorInfo(self.offset, size, ps_id))
    self.offset += size
    return dest
  
  def visit(self):
    """
    展示Chunk内所有tensor信息
    """
    for info in self.tensor_infos:
      print(f"tensor in chunk start {info.start}, end {info.start + info.size}, ps_id {info.ps_id}")

  def move(self, 
          param_dict : Dict[int, torch.nn.Parameter], 
          target_device : torch.device):
    """
    将这个chunk移动到device上
    """
    if self.device_type == target_device:
      return
    print(f'chunk move from {self.device_type} to {target_device}')
    self.payload = self.payload.to(target_device)
    # 将参数指针重新定位到新的设备上
    start = 0
    for info in self.tensor_infos:
      param = param_dict[info.ps_id]
      size = param.ps_numel
      param.ps_tensor =  self.payload.narrow(0, start, size)
      start += size
    self.device_type = target_device

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
    self.gpu_index = gpu_index
    self.data_type = data_type

    self.chunk_list = []
    self.default_chunk_size = default_chunk_size

    self.ps_manager = HybridPSManager()

    if self.ps_manager.is_init() is False:
      raise "init Manager first before use HybridPSClient"

    self.module = None
    self.ps_id = 0
    self.params_dict = {}

  def new_tensor(self, shape : torch.Size, ps_id : int = None):
    """
    在PS上新分配shape大小空间, ps_id表示param的id
    """
    numel = 1
    for elem in shape:
      numel *= elem
    
    if len(self.chunk_list) == 0:
      # 根据当前client所在设备为参考，使用manager调度获得一个最佳的device
      chunk_size = max(self.default_chunk_size, numel)
      device = self.ps_manager.schedule(chunk_size, self.gpu_index)
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = chunk_size))
      self.ps_manager.add(device.type, device.index, chunk_size)
    dest = self.chunk_list[-1].allocate(numel, ps_id)
    if dest is None:
      chunk_size = max(self.default_chunk_size, numel)
      device = self.ps_manager.schedule(chunk_size, self.gpu_index)
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = chunk_size))
      dest = self.chunk_list[-1].allocate(numel, ps_id)
      self.ps_manager.add(device.type, device.index, chunk_size)

    print(f'client new_tensor on')

    return dest.view(shape)

  @staticmethod
  def is_ps_param(parameter : torch.nn.Parameter):
    return hasattr(parameter, 'ps_id')
  
  def _convert_to_ps_param(self, param : torch.nn.Parameter):
    param.ps_id = self.ps_id # TODO(jiaruifang) generate a id
    param.ps_numel = param.numel()
    param.ps_shape = param.shape
    param.ps_tensor = None

    # 如果ps_tensor已经存在了，则将param删除
    if param.ps_tensor is not None:
      param.data = torch.ones(1).half().to(param.device)

    # 初始化ps_tensor空间
    if param.ps_tensor is None:
      param.ps_tensor = self.new_tensor(param.shape, self.ps_id)

    # 拷贝param数据到ds_tensor上
    one_dim_param = param.contiguous().view(-1)
    param.ps_tensor.copy_(one_dim_param.view(param.ps_shape))
    
    # 将原来数据删除
    param.data = torch.ones(1).half().to(param.device)
    
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
    @deprated
    Register a tensor to HybridPSClient's payload.
    Tensors are flatten and concated in a contigous memory space.
    """
    shape = src_tensor.size()
    dest = self.new_tensor(shape)
    #TODO(jiaruifang) 梯度怎么办?
    dest.data.copy_(src_tensor.data)
    src_tensor.data = dest.data

  def visit(self):
    for idx, chunk in enumerate(self.chunk_list):
      print(f"chunk {idx} on device {chunk.device_type}")
      chunk.visit()

  def chunk_move(self, chunk_id : int, device : torch.device):
    """
    测试函数，将chunk_id的chunk移动到gpu上
    需要对对应param重新赋值
    """
    print(f'client move_to_gpu {chunk_id}')
    if self.chunk_list[chunk_id].device_type != device:
      print(f'client move_to_gpu {chunk_id} again')
      self.chunk_list[chunk_id].move(self.params_dict, device)

  def free_cpu(self, size):
    """
    给cpu腾出size大小空间。
    """
    acc_free_size = 0
    for chunk in self.chunk_list:
      if chunk.device_type.type == 'cpu':
        chunk.move_to_gpu()
        acc_free_size += chunk.capacity
        if acc_free_size >= size:
          break

  def free_gpu(self, size):
    """
    给gpu腾出size大小空间。
    """
    acc_free_size = 0
    for chunk in self.chunk_list:
      if chunk.device_type == 'cuda':
        chunk.move_to_cpu()
        acc_free_size += chunk.capacity
        if acc_free_size >= size:
          break

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

