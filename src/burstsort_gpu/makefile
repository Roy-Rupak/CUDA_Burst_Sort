compile_all:
	nvcc -O3 -arch=sm_35 -rdc=true  -w -pg -DPAGING -o naskitis_copybased_burst_sort naskitis_copybased_burst_sort.cu  sort_module.o common.cu
	@cat USAGE_POLICY.txt
