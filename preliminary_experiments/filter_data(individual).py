import os
import mne
import numpy as np
import warnings

# 数据目录和新的输出目录
data_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette'
filter_output_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/filtered-data'

# 如果新的输出目录不存在，则创建
if not os.path.exists(filter_output_directory):
    os.makedirs(filter_output_directory)

# 获取所有 PSG 文件
psg_files = [f for f in os.listdir(data_directory) if 'PSG' in f]

# 提取唯一用户ID
unique_subject_ids = set()
for psg_file in psg_files:
    subject_id = psg_file.split('PSG')[1].split('.')[0][:4]  # 提取前四位作为用户ID
    unique_subject_ids.add(subject_id)
    if len(unique_subject_ids) == 10:
        break

# 仅处理前10个用户
processed_subjects = list(unique_subject_ids)[:10]

# 定义带通滤波器函数
def bandpass_filter(data, sfreq=100, l_freq=0.3, h_freq=45):
    # 创建一个 RawArray 对象
    data = data.reshape(1, -1)  # 确保数据形状为 (n_channels, n_times)
    info = mne.create_info(ch_names=['EEG'], sfreq=sfreq, ch_types=['eeg'])
    raw = mne.io.RawArray(data, info)
    
    # 应用带通滤波器
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    return raw.get_data()

# 处理每个用户的数据
for subject_id in processed_subjects:
    # 找到对应的 PSG 文件
    psg_file = f'PSG{subject_id}.edf'
    psg_path = os.path.join(data_directory, psg_file)
    
    if os.path.exists(psg_path):
        # 忽略特定警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # 读取 PSG 文件
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            data = raw.get_data()
            
            # 提取第一个通道的信号
            channel_data = data[0, :]
            
            # 对信号进行带通滤波
            filtered_data = bandpass_filter(channel_data, sfreq=raw.info['sfreq'])
            
            # 保存滤波后的数据
            output_file = os.path.join(filter_output_directory, f'filtered_{subject_id}.npy')
            np.save(output_file, filtered_data)
            
            print(f'Processed and saved filtered data for subject {subject_id}')
    else:
        print(f'PSG file for subject {subject_id} not found')

print('Filtered data processing complete.')
