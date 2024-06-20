import os
import mne
import numpy as np
import warnings



# 原始数据目录和新的输出目录
segmented_data_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/single_channel_data/EOG'
filter_output_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/filtered_single_channel_data/EOG'

# 如果新的输出目录不存在，则创建
if not os.path.exists(filter_output_directory):
    os.makedirs(filter_output_directory)

# 定义带通滤波器函数
def bandpass_filter(data, sfreq=100, l_freq=0.3, h_freq=10):
    print(data.shape)
    # 创建一个 RawArray 对象
    data = data.reshape(1, -1)  # 确保数据形状为 (n_channels, n_times)
    print(data.shape)
    info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info)
    
    # 应用带通滤波器
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    return raw.get_data()

# 加载并滤波每个阶段的数据
for stage in ['W_EOG', 'N1_EOG', 'N2_EOG', 'N3_EOG', 'REM_EOG', 'other_EOG']:
    # 构建文件路径
    input_file_path = os.path.join(segmented_data_directory, f'{stage}.npy')
    
    # 检查文件是否存在
    if os.path.exists(input_file_path):
        # 加载数据
        data = np.load(input_file_path)
        
        # 滤波数据
        filtered_data = np.array([bandpass_filter(segment, sfreq=100) for segment in data])
        
        # 保存滤波后的数据
        output_file_path = os.path.join(filter_output_directory, f'filtered_{stage}.npy')
        np.save(output_file_path, filtered_data)
        
        print(f'Filtered data for {stage} stage saved to {output_file_path}')
    else:
        print(f'{stage}.npy file not found in {segmented_data_directory}')

print('Filtered data processing complete.')
