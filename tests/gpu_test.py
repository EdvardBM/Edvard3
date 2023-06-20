import torch

# Verify if CUDA is available
print("Cuda is available: ", torch.cuda.is_available())

# Get the name of the GPU
print("GPU Device Name: ", torch.cuda.get_device_name(0))

# Create a tensor
a = torch.Tensor([1.0])

# Send the tensor to the GPU
if torch.cuda.is_available():
    a = a.cuda()

# Print out the device of the tensor
print("Tensor's device: ", a.device)
