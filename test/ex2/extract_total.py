import re

def extract_all_times(file_content):
    """
    从文件内容中提取所有以node2vec或SOPR开头的数据块的关键时间指标
    
    参数:
    file_content (str): 整个文件的内容
    
    返回:
    list: 包含所有数据块时间指标的列表，每个元素是一个三元组 (总时间, GPU时间, CPU时间)
    """
    # 使用正则表达式分隔各个数据块
    # 识别以[node2vec_numwalks_或[SOPR_numwalks_开头的块
    block_pattern = r'\[(node2vec|SOPR)_numwalks_.*?\](.*?)(?=\[(node2vec|SOPR)_numwalks_|$)'
    blocks = re.findall(block_pattern, file_content, re.DOTALL)
    
    all_times = []
    for match in blocks:
        # match[1] 包含数据块的内容（正则表达式中的第二个捕获组）
        block = match[1]
        
        # 提取当前块中的时间数据
        total_time  = None
        
        for line in block.strip().split('\n'):
            line = line.strip()
            if line.startswith('.0_Total_time='):
                total_time = float(line.split('=')[1])
        
        # 确保提取到了所有三个时间指标
        if total_time is not None :
            all_times.append((total_time))
    
    return all_times

def save_times_to_file(time_data, output_file='./test/ex2/total.csv'):
    """
    将所有时间数据保存到CSV文件
    
    参数:
    time_data (list): 包含所有数据块时间指标的列表
    output_file (str): 输出文件名，默认为'result.csv'
    """
    try:
        with open(output_file, 'w') as file:
            # 写入表头
            #file.write("Total_time\n")
            # 写入每一行数据
            for total in time_data:
                file.write(f"{total}\n")
        print(f"已成功将 {len(time_data)} 组数据写入 {output_file}")
    except Exception as e:
        print(f"写入文件时出错: {e}")

if __name__ == "__main__":
    try:
        # 从文件读取内容
        with open('./randgraph_metrics.txt', 'r') as file:
            content = file.read()
        
        # 提取所有时间数据
        all_times = extract_all_times(content)
        
        if not all_times:
            print("未找到任何完整的数据块！")
        else:
            # 输出统计信息
            print(f"共提取到 {len(all_times)} 组时间数据")
            
            # 保存到文件
            save_times_to_file(all_times)
            
            # 打印第一组数据示例
            print("\n第一组数据示例:")
            print(f"总时间: {all_times[0][0]}")
            print(f"GPU时间: {all_times[0][1]}")
            print(f"CPU时间: {all_times[0][2]}")
    
    except FileNotFoundError:
        print("错误：找不到metrics.txt文件，请确保文件在正确的路径下。")
    except Exception as e:
        print(f"发生未知错误：{e}")    