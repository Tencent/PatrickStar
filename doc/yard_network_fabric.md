## Network Topology of a node of WeChat Yard
```nvidia-smi topo -m```

GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
  
GPU0   X   NV1  NV2  NV1  SYS  SYS  SYS  NV2

GPU1  NV1   X   NV1  NV2  SYS  SYS  NV2  SYS

GPU2  NV2  NV1   X   NV2  SYS  NV1  SYS  SYS

GPU3  NV1  NV2  NV2   X   NV1  SYS  SYS  SYS

GPU4  SYS  SYS  SYS  NV1   X   NV2  NV2  NV1

GPU5  SYS  SYS  NV1  SYS  NV2   X   NV1  NV2

GPU6  SYS  NV2  SYS  SYS  NV2  NV1   X   NV1

GPU7  NV2  SYS  SYS  SYS  NV1  NV2  NV1    X

```nvidia-smi nvlink --status -i 0```

GPU 0: Tesla V100-SXM2-32GB (UUID: GPU-4b6ebbfe-8eac-8fed-1939-b4c545eafa7f)

   Link 0: 25.781 GB/s
   
   Link 1: 25.781 GB/s
   
   Link 2: 25.781 GB/s
   
   Link 3: 25.781 GB/s
   
   Link 4: 25.781 GB/s
   
   Link 5: 25.781 GB/s
