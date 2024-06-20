'''将EDF文件转为npy文件，以便后续处理'''
import mne
import numpy as np
import os
import warnings

# 数据目录和输出目录
data_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/sleep-cassette'
output_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/original_data'

# 如果输出目录不存在，则创建
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 初始化分类数据容器
W, N1, N2, N3, REM, other = [], [], [], [], [], []

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

# 定义每段的固定长度（这里假设为 3000 样本，具体根据需要调整）
fixed_length = 3000

for subject_id in processed_subjects:
    # 找到对应的 PSG 和注释文件
    psg_file = f'PSG{subject_id}.edf'
    anno_file = f'anno{subject_id}.edf'
    
    # 构建文件路径
    psg_path = os.path.join(data_directory, psg_file)
    anno_path = os.path.join(data_directory, anno_file)
    
    if os.path.exists(psg_path) and os.path.exists(anno_path):
        # 忽略特定警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # 读取 PSG 文件
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            data = raw.get_data()
            
            # 读取注释文件
            annotations = mne.read_annotations(anno_path)
            
            # 将注释添加到 raw 对象中
            raw.set_annotations(annotations)
        
        # 计算每 30 秒的样本数
        sfreq = int(raw.info['sfreq'])
        epoch_duration = 30 * sfreq
        
        # 遍历注释，根据注释分割信号数据
        for annot in annotations:
            onset = int(annot['onset'] * sfreq)
            duration = int(annot['duration'] * sfreq)
            label = annot['description']
            
            for start in range(onset, onset + duration, epoch_duration):
                end = start + epoch_duration
                if end > data.shape[1]:
                    end = data.shape[1]
                
                segment = data[:, start:end]
                
                # 确保段的长度一致
                if segment.shape[1] < fixed_length:
                    padding = np.zeros((segment.shape[0], fixed_length - segment.shape[1]))
                    segment = np.hstack((segment, padding))
                elif segment.shape[1] > fixed_length:
                    segment = segment[:, :fixed_length]
                
                if label == 'Sleep stage W':
                    W.append(segment)
                elif label == 'Sleep stage 1':
                    N1.append(segment)
                elif label == 'Sleep stage 2':
                    N2.append(segment)
                elif label == 'Sleep stage 3' or label == 'Sleep stage 4':
                    N3.append(segment)
                elif label == 'Sleep stage R':
                    REM.append(segment)
                else:
                    other.append(segment)
        
        print(f'Processed data and annotations for subject {subject_id}')
    else:
        print(f'Files for subject {subject_id} not found')

# 将分类数据保存为 numpy 数组
np.save(os.path.join(output_directory, 'W.npy'), np.array(W))
np.save(os.path.join(output_directory, 'N1.npy'), np.array(N1))
np.save(os.path.join(output_directory, 'N2.npy'), np.array(N2))
np.save(os.path.join(output_directory, 'N3.npy'), np.array(N3))
np.save(os.path.join(output_directory, 'REM.npy'), np.array(REM))
np.save(os.path.join(output_directory, 'other.npy'), np.array(other))

print('Data processing complete.')
