import pandas as pd
import numpy as np
from datetime import datetime
import os

def convert_to_recbole_format(input_file, output_file):
    """
    将图书馆借阅数据转换为RecBole格式

    Parameters:
    input_file: 输入文件路径
    output_file: 输出文件路径
    """

    # 读取数据
    df = pd.read_csv(input_file)

    # 检查数据基本信息
    print("原始数据形状:", df.shape)
    print("原始数据列名:", df.columns.tolist())
    print("\n前5行数据:")
    print(df.head())

    # 数据预处理
    # 1. 处理时间格式问题 - 还书时间和续借时间可能没有空格
    time_columns = ['借阅时间', '还书时间', '续借时间']

    for col in time_columns:
        if col in df.columns:
            # 处理缺失值
            df[col] = df[col].fillna('')
            # 格式化时间字符串（处理没有空格的情况）
            df[col] = df[col].apply(lambda x: format_time_string(x))

    # 2. 选择timestamp列：如果续借次数不为0，使用续借时间，否则使用借阅时间
    df['selected_time'] = df.apply(
        lambda row: row['续借时间'] if pd.notna(row['续借时间']) and row['续借时间'] != '' and row['续借次数'] > 0
        else row['借阅时间'],
        axis=1
    )

    # 3. 将时间转换为timestamp（浮点数）
    df['timestamp'] = df['selected_time'].apply(convert_to_timestamp)

    # 4. 创建RecBole格式的数据框
    recbole_df = pd.DataFrame({
        'user_id:token': df['user_id'],
        'item_id:token': df['book_id'],
        'timestamp:float': df['timestamp']
    })

    # 5. 删除timestamp为NaN的行（时间格式错误的数据）
    original_count = len(recbole_df)
    recbole_df = recbole_df.dropna(subset=['timestamp:float'])
    cleaned_count = len(recbole_df)

    print(f"\n数据清理: 原始{original_count}行, 清理后{cleaned_count}行, 删除{original_count - cleaned_count}行")

    # 6. 按时间排序
    recbole_df = recbole_df.sort_values('timestamp:float')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 7. 保存为RecBole格式的文本文件
    recbole_df.to_csv(output_file, sep='\t', index=False)

    # 输出统计信息
    print(f"\n转换完成! 输出文件: {output_file}")
    print(f"用户数: {recbole_df['user_id:token'].nunique()}")
    print(f"物品数: {recbole_df['item_id:token'].nunique()}")
    print(f"交互数: {len(recbole_df)}")
    print(
        f"时间范围: {datetime.fromtimestamp(recbole_df['timestamp:float'].min())} 到 {datetime.fromtimestamp(recbole_df['timestamp:float'].max())}")

    print("\n转换后的前5行数据:")
    print(recbole_df.head())

    return recbole_df


def format_time_string(time_str):
    """
    格式化时间字符串，处理日期和时间之间没有空格的情况
    """
    if pd.isna(time_str) or time_str == '':
        return ''

    time_str = str(time_str).strip()

    # 如果日期和时间之间没有空格，添加空格
    # 格式如: "2022-08-0811:14:55" -> "2022-08-08 11:14:55"
    if len(time_str) > 10 and time_str[10] != ' ':
        time_str = time_str[:10] + ' ' + time_str[10:]

    return time_str


def convert_to_timestamp(time_str):
    """
    将时间字符串转换为timestamp（浮点数）
    """
    if pd.isna(time_str) or time_str == '':
        return np.nan

    try:
        # 尝试解析时间格式
        dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        return float(dt.timestamp())
    except ValueError:
        try:
            # 如果上面的格式失败，尝试其他常见格式
            dt = datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
            return float(dt.timestamp())
        except ValueError:
            print(f"时间格式解析错误: {time_str}")
            return np.nan


# 使用示例
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "data", "inter_preliminary.csv")
    output_file = os.path.join(current_dir, "dataset", "data", "data.inter")

    # 执行转换
    recbole_data = convert_to_recbole_format(input_file, output_file)