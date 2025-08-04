import os
import torch
import torch.distributed as dist
import ray
import socket

@ray.remote(num_gpus=1)
def test_nccl(rank, world_size):
    # 关键配置
    os.environ.update({
        "MASTER_ADDR": "10.14.4.8",
        "MASTER_PORT": "29500",
        "NCCL_SOCKET_IFNAME": "eno1",
        "NCCL_ALGO": "ring",
        "NCCL_PROTO": "Simple",
        "NCCL_DEBUG": "INFO",
        "CUDA_LAUNCH_BLOCKING": "1"  # 同步CUDA错误报告
    })

    # 安全的设备绑定
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device_id = local_rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
        
        # 测试通信
        tensor = torch.tensor([rank], device=f"cuda:{device_id}")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[{socket.gethostname()}] Rank {rank} success: {tensor}")
        return True
    except Exception as e:
        print(f"[{socket.gethostname()}] Rank {rank} failed: {str(e)}")
        return False

if __name__ == "__main__":
    # 初始化Ray（自动检测集群）
    ray.init(address="auto", ignore_reinit_error=True)
    
    # 动态获取可用GPU数量
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    visible_gpus = torch.cuda.device_count()
    actual_gpus = min(total_gpus, visible_gpus)
    
    if actual_gpus > 0:
        print(f"Using {actual_gpus} GPUs (Total: {total_gpus}, Visible: {visible_gpus})")
        results = ray.get([test_nccl.remote(i, actual_gpus) for i in range(actual_gpus)])
        print("All tasks passed:", all(results))
    else:
        print("No available GPUs!")