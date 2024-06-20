import numpy as np
import os

# 数据目录和新的输出目录
original_output_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/original_data'
new_output_directory = 'F:/sleep-edf-database-expanded-1.0.0/sleep-edf-database-expanded-1.0.0/single_channel_data/EOG'

# 如果新的输出目录不存在，则创建
if not os.path.exists(new_output_directory):
    os.makedirs(new_output_directory)

# 读取原始数据
W = np.load(os.path.join(original_output_directory, 'W.npy'))
N1 = np.load(os.path.join(original_output_directory, 'N1.npy'))
N2 = np.load(os.path.join(original_output_directory, 'N2.npy'))
N3 = np.load(os.path.join(original_output_directory, 'N3.npy'))
REM = np.load(os.path.join(original_output_directory, 'REM.npy'))
other = np.load(os.path.join(original_output_directory, 'other.npy'))

# 提取EOG通道（假设它是第三个通道）
EOG_idx = 2  # 假设EOG是第三个通道，索引为2

# 提取EOG通道的数据
W_EOG = W[:, EOG_idx, :]
N1_EOG = N1[:, EOG_idx, :]
N2_EOG = N2[:, EOG_idx, :]
N3_EOG = N3[:, EOG_idx, :]
REM_EOG = REM[:, EOG_idx, :]
other_EOG = other[:, EOG_idx, :]

print('Single-channel data processing complete.')

# 将新的数据保存到新的输出目录中
np.save(os.path.join(new_output_directory, 'W_EOG.npy'), W_EOG)
np.save(os.path.join(new_output_directory, 'N1_EOG.npy'), N1_EOG)
np.save(os.path.join(new_output_directory, 'N2_EOG.npy'), N2_EOG)
np.save(os.path.join(new_output_directory, 'N3_EOG.npy'), N3_EOG)
np.save(os.path.join(new_output_directory, 'REM_EOG.npy'), REM_EOG)
np.save(os.path.join(new_output_directory, 'other_EOG.npy'), other_EOG)
