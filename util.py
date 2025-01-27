import numpy as np
from scipy.linalg import eigh
def calculate_log_variance(epochs):
    """计算每个epoch的对数方差特征"""
    return np.log(np.var(epochs, axis=2))
def func_csp(dat, nPatterns=3, cov_method='normal', score_method='eigenvalue', policy='normal'):
    """
    计算共同空间模式 (Common Spatial Patterns, CSP)。

    Args:
        dat (dict): 包含数据的字典，需要包含以下键：
                    'x': np.ndarray, 形状为 (n_trials, n_channels, n_times) 的数据。
                    'y_logic': np.ndarray, 形状为 (n_classes, n_trials) 的逻辑标签。
                    'class': list, 包含类别名称的列表。
        nPatterns (int): 要提取的 CSP 模式的数量 (每个类别提取的数量)。默认为 3。
        cov_method (str): 计算协方差矩阵的方法，'normal' 或 'average'。默认为 'normal'。
        score_method (str): CSP 得分的计算方法，目前只实现了 'eigenvalue'。默认为 'eigenvalue'。
        policy (str) : CSP 滤波器选择方式，'normal' 或 'directorscut'。默认为 'normal'。

    Returns:
        tuple: 包含以下内容的元组：
               - dat_csp (dict): 经过 CSP 滤波后的数据。
               - CSP_W (np.ndarray): CSP 滤波器，形状为 (n_channels, 2 * nPatterns)。
               - CSP_D (np.ndarray): CSP 得分，形状为 (2 * nPatterns,)。
    """

    X = dat['x']
    y_logic = dat['y_logic']
    classes = dat['class']

    n_trials, n_channels, n_times = X.shape
    n_classes = len(classes)

    if score_method != 'eigenvalue':
        raise NotImplementedError("目前只实现了 'eigenvalue' 得分计算方法。")

    # 1. 计算每个类别的协方差矩阵
    reg_param = 1e-6
    R = np.zeros((n_channels, n_channels, n_classes))
    if cov_method == 'normal':
        for i in range(n_classes):
            idx = np.where(y_logic[i, :])[0]
            X_class = X[idx]
            # 将数据重塑为 (n_samples, n_channels)，其中 n_samples = n_times * n_trials
            X_class_reshaped = X_class.transpose(1,0,2).reshape(n_channels, -1)
            R[:, :, i] = np.cov(X_class_reshaped)+ reg_param * np.eye(n_channels)
    elif cov_method == 'average':
        for i in range(n_classes):
            idx = np.where(y_logic[i, :])[0]
            C = np.zeros((n_channels, n_channels))
            for m in idx:
                C += np.cov(X[m].T) 
            R[:, :, i] = C / len(idx)
    else:
        raise ValueError("未知的协方差矩阵计算方法。")

    # 2. 计算 CSP 滤波器和得分
    # 使用广义特征值分解
    D, W = eigh(R[:, :, 1], R[:, :, 0] + R[:, :, 1])

    # 根据 policy 选择 CSP 滤波器
    if policy == 'normal':
      CSP_W = W[:, np.r_[0:nPatterns, -nPatterns:0]]
      CSP_D = D[np.r_[0:nPatterns, -nPatterns:0]]
    elif policy == 'directorscut':
      score = D
      absscore = 2 * (np.maximum(score, 1 - score) - 0.5)
      sorted_indices = np.argsort(score)
      Nh = n_channels // 2
      iC1 = np.where(np.isin(sorted_indices, np.arange(Nh)))[0]
      iC2 = np.flip(np.where(np.isin(sorted_indices, np.arange(n_channels - Nh, n_channels)))[0])
      iCut = np.where(absscore[sorted_indices] >= 0.66 * np.max(absscore))[0]
      idx1 = np.concatenate([[iC1[0]], np.intersect1d(iC1[1:nPatterns], iCut)])
      idx2 = np.concatenate([[iC2[0]], np.intersect1d(iC2[1:nPatterns], iCut)])
      fi = sorted_indices[np.concatenate([idx1, np.flip(idx2)])]
      CSP_W = W[:, fi]
      CSP_D = score[fi]
    else:
      raise ValueError("未知的 CSP 滤波器选择方式")

    # 3. 对数据进行 CSP 滤波
    dat_csp = {'x': np.dot(X.transpose(0,2,1), CSP_W).transpose(0,2,1),
               'y_logic': dat['y_logic'],
               'class': dat['class'],
               'fs': dat['fs'] if 'fs' in dat else None,
               't': dat['t'] if 't' in dat else None,
               'chan': dat['chan'] if 'chan' in dat else None}
    

    return dat_csp, CSP_W, CSP_D

def func_projection(dat, w):
    """
    将数据投影到指定的一组基向量上。

    Args:
        dat:  输入数据。可以是一个 NumPy 数组，也可以是一个包含 'x' 键的字典。
              如果是一个 NumPy 数组，它可以是二维 (n_samples, n_features) 或三维 (n_trials, n_channels, n_samples)。
              如果是一个字典，它应该包含一个 'x' 键，其值为 NumPy 数组。
        w:  投影矩阵，形状为 (n_features, n_components)。

    Returns:
        投影后的数据。如果输入 dat 是一个 NumPy 数组，则返回投影后的 NumPy 数组；
        如果输入 dat 是一个字典，则返回一个字典，其中 'x' 键的值被更新为投影后的数据。
    """

    if isinstance(dat, dict):
        if 'x' in dat:
            tDat = dat['x']
        else:
            raise KeyError("dat 字典必须包含 'x' 键")
    else:
        tDat = dat

    if tDat.ndim == 2:
        # 二维数据 (n_samples, n_features)
        out_data = np.dot(tDat, w)
    elif tDat.ndim == 3:
        # 三维数据 (n_trials, n_channels, n_samples)
        n_trials, n_channels, n_samples = tDat.shape
        # 重塑为 (n_trials * n_samples, n_channels)
        tDat_reshaped = tDat.transpose(0, 2, 1).reshape(n_trials * n_samples, n_channels)
        # 投影
        projected_data = np.dot(tDat_reshaped, w)
        # 重塑回 (n_trials, n_channels, n_components)
        out_data = projected_data.reshape(n_trials, n_samples, -1).transpose(0, 2, 1)
    else:
        raise ValueError("dat['x'] 必须是二维或三维数组")

    if isinstance(dat, dict):
        out = dat.copy()
        out['x'] = out_data
        return out
    else:
        return out_data