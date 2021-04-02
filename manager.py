import torch
from torch.multiprocessing import Process, Manager

######### Global Scheduler ###########
class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class HybridPSManager(metaclass = SingletonMeta):
  def __init__(self):
    mp_manager = Manager()
    self._is_init_ = mp_manager.Value('_is_init_', False)
    self.gpu_max_mem_list = mp_manager.list([])
    self.cpu_max_mem_list = mp_manager.list([])
    self.gpu_used_mem_list = mp_manager.list([])
    self.cpu_used_mem_list = mp_manager.list([])

  def init(self, gpu_info, cpu_info):
    """
    知道所有设备的使用情况，来指导payload的迁移
    singleton类，被所有进程访问
    """
    for item in gpu_info:
      self.gpu_max_mem_list.append(item)
      self.gpu_used_mem_list.append(0)

    for item in cpu_info:
      self.cpu_max_mem_list.append(item)
      self.cpu_used_mem_list.append(0)
    self._is_init_.value = True

  def reset(self, gpu_info, cpu_info):
    mp_manager = Manager()
    self._is_init_ = mp_manager.Value('_is_init_', False)
    self.gpu_max_mem_list = mp_manager.list([])
    self.cpu_max_mem_list = mp_manager.list([])
    self.gpu_used_mem_list = mp_manager.list([])
    self.cpu_used_mem_list = mp_manager.list([])
    self.init(gpu_info, cpu_info)
    
  def is_init(self):
    return self._is_init_.value
  
  def visit(self):
    for idx, value in enumerate(self.gpu_used_mem_list):
      print(f"GPU:{idx} used mem {value}")
    for idx, value in enumerate(self.cpu_used_mem_list):
      print(f"CPU:{idx} used mem {value}")

  def add(self, device_type, index, size):
    if index is None:
      index = 0
    
    if device_type == "cpu":
      self.cpu_used_mem_list[index] += size
    elif device_type == "cuda":
      self.gpu_used_mem_list[index] += size
    else:
      raise f"device type {device_type} is not supported"

  def delete(self, device_type, index, size):
    if index is None:
      index = 0

    if device_type == "cpu":
      self.cpu_used_mem_list[index] -= size
    elif device_type == "cuda":
      self.gpu_used_mem_list[index] -= size
    else:
      raise f"device type {device_type} is not supported"

  def migrate_out(self, device_type, index, size):
    """
    找到另一个设备
    返回 device_type和index
    """
    pass

  def schedule(self, size, refer_dev_idx):
    """
    找到一个设备，可以分配size大小存储空间
    """
    if self.avaiable_mem("cpu", refer_dev_idx) >= size:
      return torch.device("cpu")
    elif self.avaiable_mem("cuda", refer_dev_idx) >= size:
      return torch.device(f"cuda:{refer_dev_idx}")
    else:
      for idx in range(self.gpu_num()):
        if idx == refer_dev_idx:
          pass
        if self.avaiable_mem("cuda", idx) >= size:
          self.add("cuda", idx, size)
          return torch.device(f"cuda:{idx}")
    raise f"HybridPSManager can not find {size} space"

  def avaiable_mem(self, device_type, index):
    if device_type == "cuda":
      return self.gpu_max_mem_list[index] - self.gpu_used_mem_list[index]
    elif device_type == "cpu":
       return self.cpu_max_mem_list[index] - self.cpu_used_mem_list[index]
  
  def gpu_num(self):
    return len(self.gpu_max_mem_list)
  
  def cpu_num(self):
    return len(self.cpu_max_mem_list)

  def used_mem(self, device_type, index):
    if device_type == "cpu":
      return self.cpu_used_mem_list[index]
    elif device_type == "cuda":
      return self.gpu_used_mem_list[index]
  
  def max_mem(self, device_type, index):
    if device_type == "cpu":
      return self.cpu_max_mem_list[index]
    elif device_type == "cuda":
      return self.cpu_max_mem_list[index]

if __name__ == "__main__":
  s1 = HybridPSManager()
  s1.init([64, 64], [128])

  # do nothing if you initialize a singleton twice
  s2 = HybridPSManager()
  s2.init([32, 32, 3], [32])
  assert s2.gpu_num() == 2

  if id(s1) == id(s2):
      print("HybridPSManager works, both variables contain the same instance.")
  else:
      print("HybridPSManager failed, variables contain different instances.")