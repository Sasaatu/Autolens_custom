import torch

if torch.cuda.is_available():
    print("GPUが利用可能です。")
    print("CUDA version:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    print("GPU name:", torch.cuda.get_device_name(0)) # 最初のGPUの名前を表示
else:
    print("GPUが利用できません。")