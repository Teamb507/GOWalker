import re

def extract_all_times(file_content):
    block_pattern = r'\[(node2vec|SOPR)_numwalks_.*?\](.*?)(?=\[(node2vec|SOPR)_numwalks_|$)'
    blocks = re.findall(block_pattern, file_content, re.DOTALL)
    
    all_times = []
    for match in blocks:
        block = match[1]
        total_time  = None
        
        for line in block.strip().split('\n'):
            line = line.strip()
            if line.startswith('.0_Total_time='):
                total_time = float(line.split('=')[1])
        if total_time is not None :
            all_times.append((total_time))
    
    return all_times

def save_times_to_file(time_data, output_file='./test/experiment2/total.csv'):
    try:
        with open(output_file, 'w') as file:
            for total in time_data:
                file.write(f"{total}\n")

if __name__ == "__main__":
    try:
        with open('./randgraph_metrics.txt', 'r') as file:
            content = file.read()
        all_times = extract_all_times(content)
        
        if not all_times:
            print("can't find data!")
        else:
            save_times_to_file(all_times)