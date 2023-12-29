cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project (myprojec)
enable_language(CUDA)
include_directories ("${PROJECT_SOURCE_DIR}/third_party/eigen")
# 查找当前目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)
# 添加 math 子目录
add_subdirectory(src)
# 指定生成目标 
add_library(gpu_conv_lib STATIC src/layer/gpu-new-forward-basic.cu src/layer/gpu-new-forward.h src/layer/gpu-utils.cu src/layer/gpu-utils.h )

include_directories(${CUDA_INCLUDE_DIRS})

add_executable(demo ${DIR_SRCS})
# 添加链接库
target_link_libraries(demo gpu_conv_lib MiniDNNLib ${CUDA_LIBRARIES})
