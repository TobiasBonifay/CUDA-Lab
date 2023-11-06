NVCC = /usr/local/cuda/bin/nvcc
# Use the next line for Maxwell architecture
#CUDA_FLAGS = -gencode arch=compute_52,code=sm_52 
# Use the next line for Pascal architecture
#CUDA_FLAGS = -gencode arch=compute_61,code=sm_61 
# Use the next line for Pascal with integrated GPU (Jetson TX2)
# CUDA_FLAGS = -gencode arch=compute_62,code=sm_62
# Use the next line for Ampere architecture
CUDA_FLAGS = -gencode arch=compute_86,code=sm_86

which-device:  which-device.cu
	$(NVCC) $(CUDA_FLAGS) $< -o $@

timing: timing.cu
	$(NVCC) $(CUDA_FLAGS) $< -o $@

vectorAdd: vectorAdd.cu
	$(NVCC) $(CUDA_FLAGS) $< -o $@

all = vectorAdd timing which-device

clean: 
	rm -rf *o $(TARGET)
