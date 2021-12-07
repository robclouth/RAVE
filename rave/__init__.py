import GPUtil as gpu
from os import environ
import torch


def is_gpu_available():
    CUDA = gpu.getAvailable(maxMemory=.05)
    if len(CUDA):
        environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    return use_gpu