#!/bin/bash

# 设置要读取的文件夹路径
YML_DIR="/data/haiyu/workspace/AverNet/options/test"

# 遍历文件夹中的所有 .yml 文件
for yml_file in "$YML_DIR"/*.yml; do
    # 检查是否存在 .yml 文件
    if [ -f "$yml_file" ]; then
        # 运行指定的命令
        CUDA_VISIBLE_DEVICES=2 python basicsr/test.py -opt "$yml_file"
    else
        echo "No .yml files found in the directory."
    fi
done
