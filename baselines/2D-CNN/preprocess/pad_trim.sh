#!/bin/bash

file_name=$1
max_file_size=$2

# 检查参数是否为空
if [ $# -ne 2 ]; then
    echo "Usage: $0 <file_name> <max_file_size>"
    exit 1
fi

# 检查文件是否存在
if [ ! -e "$file_name" ]; then
    echo "Error: File $file_name not found."
    exit 1
fi

# 获取文件大小
file_size=$(wc -c < "$file_name")

# 判断文件大小是否大于最大文件大小
if [ "$file_size" -gt "$max_file_size" ]; then
    # 截断文件大小到最大文件大小
    truncate -s "$max_file_size" "$file_name"
elif [ "$file_size" -lt "$max_file_size" ]; then
    # 在文件后面pad字节0直到文件达到最大文件大小
    pad_size=$((max_file_size - file_size))
    dd if=/dev/zero bs=1 count="$pad_size" >> "$file_name"
fi


