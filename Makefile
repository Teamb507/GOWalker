# 编译器和参数
NVCC = nvcc
SRC_NODE2VEC = apps/Node2vec.cu    # 分离源文件
SRC_S = apps/SOPR.cu                      # 新增单独变量
TARGET_NODE2VEC = apps/Node2vec     # 分离目标
TARGET_S = apps/SOPR                       # 新增单独目标

CUDA_PATH = /usr/local/cuda-12.2

# 编译参数
NVCC_FLAGS = -I. -Xcompiler -fopenmp -w  

# 头文件依赖目录（所有目标共享）
INCLUDE_DIRS = api engine logger util metrics preprocess gpu
DEPS = $(foreach dir, $(INCLUDE_DIRS), $(wildcard $(dir)/*.hpp))

# 默认目标：构建所有可执行文件
all: $(TARGET_NODE2VEC) $(TARGET_S)

# 为每个目标单独定义规则（关键修改）
$(TARGET_NODE2VEC): $(SRC_NODE2VEC) $(DEPS)
	@echo "Compiling gpu_node2vec..."
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(LIBSTD_PATH): $(NVCC) $(NVCC_FLAGS) -o $@ $<  
$(TARGET_S): $(SRC_S) $(DEPS) 	
	@echo "Compiling S..." 	
	LD_LIBRARY_PATH=$(CUDA_PATH)/lib64:$(LIBSTD_PATH): $(NVCC) $(NVCC_FLAGS) -o $@ $<
	
# 清理编译产物
clean:
	rm -f $(TARGET_NODE2VEC) $(TARGET_S)