NVCC = nvcc
SRC_NODE2VEC = apps/Node2vec.cu   
SRC_S = apps/SOPR.cu                  
TARGET_NODE2VEC = apps/Node2vec    
TARGET_S = apps/SOPR                

CUDA_PATH = /usr/local/cuda-12.2
NVCC_FLAGS = -I. -Xcompiler -fopenmp -w  
INCLUDE_DIRS = api engine logger util metrics preprocess gpu
DEPS = $(foreach dir, $(INCLUDE_DIRS), $(wildcard $(dir)/*.hpp))
all: $(TARGET_NODE2VEC) $(TARGET_S)
$(TARGET_NODE2VEC): $(SRC_NODE2VEC) $(DEPS)
	@echo "Compiling gpu_node2vec..."
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(LIBSTD_PATH): $(NVCC) $(NVCC_FLAGS) -o $@ $<  
$(TARGET_S): $(SRC_S) $(DEPS) 	
	@echo "Compiling S..." 	
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(LIBSTD_PATH): $(NVCC) $(NVCC_FLAGS) -o $@ $<
clean:
	rm -f $(TARGET_NODE2VEC) $(TARGET_S)