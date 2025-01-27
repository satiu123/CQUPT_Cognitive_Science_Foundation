import numpy as np
import scipy.io
from scipy.signal import decimate,resample
from scipy.signal import butter, filtfilt
def load_data(file):
    data=scipy.io.loadmat(file)
    return data['EEG_MI_train'][0,0],data['EEG_MI_test'][0,0]
def prep_data(data,field,fs=1000):
    new_data={}
    for item in field:
        new_data[item]=data[item]
    #降采样
    if fs<data['fs']:
        downsample_factor = int(data['fs'] // fs)
        signal=data['x'].T
        downsampled_signal = np.zeros((signal.shape[0], signal.shape[1] // downsample_factor))
        for i in range(signal.shape[0]):
            # downsampled_signal[i,:] = decimate(signal[i,:], downsample_factor, ftype='iir', zero_phase=True)
            downsampled_signal[i,:] = resample(signal[i,:], signal.shape[1] // downsample_factor)
        new_data['x']=downsampled_signal.T
        new_data['fs']=fs
        new_data['t']=data['t']//downsample_factor
    else:
        new_data['fs']=int(data['fs'])  
    return new_data
def prep_selectChannels(data, channels):
    out={}
    chan_list=data['chan'].tolist()
    for key in data.keys():
        if key not in ['x','chan']:
            out[key]=data[key]
    #查找channels在data['chan']中的索引
    all_channels_flat = [item[0] for item in chan_list[0]]  # 将 all_channels 扁平化为一维列表
    indices = []
    for channel in channels:
        index = all_channels_flat.index(channel)
        indices.append(index)
    out['x']=data['x'][:,indices]
    out['chan']=channels

    return out
def prep_bandFilter(data,band,fs):
    """
    Args:
        data (dict):, 包含有键 'x' 的数据字典
        band (list): 二维列表，频带的下限和上限频率
        fs (int): 采样率
    """
    if isinstance(data, dict):
        if 'x' in data:
          tDat = data['x']
        else:
          print("Data structure must have key 'x' for data matrix")
          return
    else:
        tDat = data

    band = np.array(band)
    nyquist_freq = 0.5 * fs
    normalized_band = band / nyquist_freq
    # 5阶巴特沃斯数字滤波器进行带通滤波
    b, a = butter(5, normalized_band, btype='band')

    if tDat.ndim == 1:
        filtered_data = filtfilt(b, a, tDat)
    elif tDat.ndim >1:
        filtered_data = filtfilt(b, a, tDat, axis=0)
    
    if isinstance(data, dict):
        data['x'] = filtered_data
        return data
    else:
      return filtered_data
def prep_segmentation(data,interval:list,fs:int):
    '''
    Args:
        data (dict): 包含数据的字典，需要包含以下键：
                    'x': np.ndarray, 形状为 (n_trials, n_channels, n_times) 的数据。
                    't': np.ndarray, 形状为 (1, n_trials) 的时间标签。
        interval (list): 脑电数据的分割区间
        fs (int): 采样率
    Returns:
        data['x']: np.array, 事件 x 通道 x 采样点
    '''
    tDat = data['x']
    t = data['t']
    ival = np.array(interval)
    idc = np.arange(np.floor(ival[0] * fs / 1000), np.ceil(ival[1] * fs / 1000) + 1, dtype=int)
    T = len(idc)
    nEvents = t.shape[1]
    nChans = tDat.shape[1]

    IV = (idc.reshape(-1, 1) @ np.ones((1, nEvents))) + (np.ones((T, 1)) @ t)
    IV = IV.astype(int)

    dat_x = tDat[IV, :]

    dat_x = np.transpose(dat_x, (1, 2, 0)) # 事件 x 通道 x 采样点
    dat_ival = np.linspace(ival[0], ival[1], T)

    new_data=data.copy()
    new_data['x'] = dat_x
    new_data['ival'] = dat_ival
    return new_data