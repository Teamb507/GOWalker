import re

def extract_all_times(file_content):
    block_pattern = r'\[(node2vec|SOPR)_numwalks_.*?\](.*?)(?=\[(node2vec|SOPR)_numwalks_|$)'
    blocks = re.findall(block_pattern, file_content, re.DOTALL)
    
    all_times = []
    for match in blocks:
        block = match[1]
        transfer_time = gpu_time = cpu_time = None
        
        for line in block.strip().split('\n'):
            line = line.strip()
            if line.startswith('.1.1_tranfer='):
                transfer_time = float(line.split('=')[1])
            elif line.startswith('.1.3_GPU_update='):
                gpu_time = float(line.split('=')[1])
            elif line.startswith('.2_CPUTime='):
                cpu_time = float(line.split('=')[1])
        if transfer_time is not None and gpu_time is not None and cpu_time is not None:
            all_times.append((transfer_time,gpu_time,cpu_time))
    
    return all_times

def save_times_to_file(time_data, output_file='./test/experiment2/nopipeline.csv'):
    try:
        with open(output_file, 'w') as file:
            for transfer,gpu,cpu in time_data:
                file.write(f"{total},{gpu},{cpu}\n")

if __name__ == "__main__":
    try:
        with open('./randgraph_metrics.txt', 'r') as file:
            content = file.read()
        all_times = extract_all_times(content)
        
        if not all_times:
            print("can't find data！")
        else:
            save_times_to_file(all_times)   