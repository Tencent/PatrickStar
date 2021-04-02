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
    
    # 根据当前client所在设备为参考，使用manager调度获得一个最佳的device
    device = self.ps_manager.schedule(numel, self.index)
    print(f'client new_tensor on {device}')
    if len(self.chunk_list) == 0:
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = max(self.default_chunk_size, numel)))
    dest = self.chunk_list[-1].allocate(numel)
    if dest is None:
      self.chunk_list.append(Chunk(device_type = device,
                                   capacity = max(self.default_chunk_size, numel)))
      dest = self.chunk_list[-1].allocate(numel)
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

  def swap_out(self):
    """
    The device has to allocate memory for more important tensor.
    The payload should migrate to the other device.
    需要和全局调度器通信来决定迁移到哪
    由于一个process可以看到一块cpu和一块gpu的存储空间
    swap也仅限于cpu和gpu之间通信
    未来可以加上gpu p2p
    """
    pass

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

