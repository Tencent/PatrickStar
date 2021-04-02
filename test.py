from manager import HybridPSManager
from client import HybridPSClient
import torch
import torch.distributed as dist
from common import distributed_test

manager = HybridPSManager()

@distributed_test(world_size=1)
def test_client():
  world_size = dist.get_world_size()
  manager.init([32] * world_size, [64])
  local_rank = dist.get_rank()

  # 申请两个tensor
  param1 = torch.randn(40, device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
  param2 = torch.randn(15, device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))

  # 用一个HybridPSClient来管理这两个tensor
  # GPU Chunk 40, 20
  client = HybridPSClient(index = local_rank, 
                          default_chunk_size = 20)

  client.register_tensor(param1)
  client.register_tensor(param2)

  assert param1.device == torch.device('cpu')
  assert param2.device == torch.device('cpu')

  # 申请第三个tensor，此时cpu内存不足，被放在gpu上
  # GPU Chunk 40, 20 CPU Chunk 20
  param3 = torch.randn(20, device = torch.device('cpu'))
  client.register_tensor(param3)
  assert param3.device == torch.device('cuda:0')

  # 申请第四个tensor, 需要chunk size=20大小, GPU没有空间了，会跑出异常
  except_flag = False
  try:
    param4 = client.new_tensor((1, 5))
  except:
    except_flag = True
  assert(except_flag)

def test_mgr_dist():
  # 在两个进程上使用HybridPSClient，测试manager效果
  manager.init([32, 32], [64])

  @distributed_test(world_size=2)
  def test_dist_init():
      assert dist.is_initialized()
      assert dist.get_world_size() == 2
      assert dist.get_rank() < 2
      print("pass test_init")
  
  #测试mgr正确更新
  def test_mgr_update():
    manager = HybridPSManager()

    @distributed_test(world_size=2)
    def test_add():
      local_rank = dist.get_rank()
      manager.add("cuda", local_rank, (local_rank+1) * 10)
      manager.add("cuda", local_rank, (local_rank+1) * 22)

    @distributed_test(world_size=2)
    def test_delete():
      local_rank = dist.get_rank()
      if local_rank == 0:
        manager.delete("cuda", local_rank, 10)

    test_add()
    assert(manager.used_mem("cuda", 0) == 32)
    assert(manager.used_mem("cuda", 1) == 64)
    assert(manager.used_mem("cpu", 0) == 0)
  
    test_delete()
    assert(manager.used_mem("cuda", 0) == 22)
    assert(manager.used_mem("cuda", 1) == 64)

  test_dist_init()
  test_mgr_update()

if __name__ == "__main__":
  test_client()
  test_mgr_dist()