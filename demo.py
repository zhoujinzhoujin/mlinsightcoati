import mlinsight
import torch

def cuMemAllocCB(ptr,size):
    print("cuMemAlloc Finished",ptr,size);

def cuMemFreeCB(ptr):
    print("cuMemFree Finished",ptr);


mlinsight.install(cuMemAllocCB,cuMemFreeCB)

tensor = torch.tensor([3,4,5], dtype=torch.int64)
cuda_device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
tensor.to(cuda_device)
del tensor
torch.cuda.empty_cache()