import torch
from torch.multiprocessing import Process, Manager

from manager import HybridPSManager

############ CHUNK ###################

class TensorInfo(object):
  def __init__(self, start, size):
    self.start = start
    self.size = size
    
class Chunk(object):
  def __init__(self, capacity = 100, device_type = torch.device('cpu')):
    """
    Chunk是数据迁移的最小单位，
    它用一段连续的内存来存储张量
    TODO(jiaruifang) 不存在释放tensor内存的问题，只有temp buff需要释放
    HybridPSClient管理的tensor只有迁移
    """
    self.capacity = capacity
    self.offset = 0
    self.device_type = device_type
    self.payload = torch.zeros(capacity, dtype = torch.float, device = device_type)

    self.tensor_infos = []
      
  def allocate(self, size):
    if self.capacity - self.offset < size:
      return None
    dest = self.payload.narrow(0, self.offset, size)
    self.tensor_infos.append(TensorInfo(self.offset, size))
    self.offset += size
    return dest
  
  def visit(self):
    for info in self.tensor_infos:
      print(f"tensor in chunk start {info.start}, end {info.start + info.size}")

  def move_to_gpu(self):
    if self.device.type == 'gpu':
      return
    self.payload = self.payload.to(torch.cuda.current_device())

  def move_to_cpu(self):
    if self.device.type == 'cpu':
      return
    self.payload = self.payload.cpu()
######### HybridPS #############

class HybridPSClient(object):
  def __init__(self,
                index = 0, 
                data_type = torch.float,
                default_chunk_size = 64):
    """
    管理一个Process的Param, AccGrad, OS数据。
    每个进程管理一个GPU
    在DeepSpeed中每个Process可以看到一块gpu和cpu的内存。
    功能:
      1. 充分利用cpu和gpu内存
      2. 细粒度调度，HybridPSClient包含若干chunk
    """
    # device can be a cpu or a gpu
    self.index = index
    self.data_type = data_type

    self.chunk_list = []
    self.default_chunk_size = default_chunk_size


    self.ps_manager = HybridPSManager()
    # TODO(jiaruifang) 确保manager已经初始化完毕

  def new_tensor(self, shape):
    numel = 1
    for elem in shape:
      numel *= elem
    
    if len(self.chunk_list) == 0:
      # 根据当前client所在设备为参考，使用manager调度获得一个最佳的device
      chunk_size = max(self.default_chunk_size, numel)
      device = self.ps_manager.schedule(chunk_size, self.index)
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = chunk_size))
      self.ps_manager.add(device.type, device.index, chunk_size)
    dest = self.chunk_list[-1].allocate(numel)
    if dest is None:
      chunk_size = max(self.default_chunk_size, numel)
      device = self.ps_manager.schedule(chunk_size, self.index)
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = chunk_size))
      dest = self.chunk_list[-1].allocate(numel)
      self.ps_manager.add(device.type, device.index, chunk_size)
      
    print(f'client new_tensor on {device}')
    return dest.view(shape)

  def register_tensor(self, src_tensor):
    """
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

  def generate(self):
    """
    Generate all tensor in payload for visit.
    """
    # for i in self.tensor_num:
    #   yield self.payload.narrow()
    pass

  def free_cpu(self, size):
    """
    给cpu腾出size大小空间。
    """
    acc_free_size = 0
    for chunk in self.chunk_list:
      if chunk.device_type == 'cpu':
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

  def broadcast(self, local_tensor):
    """
    必须所有process同时执行，规约后的payload存储在哪由调度器决定
    """
    pass

