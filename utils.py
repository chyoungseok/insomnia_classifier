import h5py
import numpy as np
import torch


def add_zero_to_id(sub_id):
    i = 1000
    while 1:
        if (sub_id // i) >= 1:
            add_zeros = '0' * (3 - int(np.log10(i)))
            return f"{add_zeros}{sub_id}"
        else:
            i /= 10

def load_single_ppg(path_ppg):
    with h5py.File(path_ppg, "r") as hdf:
        return hdf["data"][:]
    
def pad_and_create_mask(list_data, padding_value=0, verobose=False):
    """
    데이터의 길이를 max_length에 맞춰 zero-padding을 적용하고,
    패딩 여부를 나타내는 마스크를 생성합니다.

    Args:
        list_data (list of tensors): 입력 신호 리스트 (각 샘플은 서로 다른 길이를 가질 수 있음)
        max_length (int): 패딩 후 고정 길이
        padding_value (int): 패딩에 사용할 값 (기본값: 0)
    
    Returns:
        padded_list_data (torch.Tensor): 패딩된 데이터
        mask (torch.Tensor): 패딩 여부를 나타내는 마스크
    """

    _len_data = []
    for data in list_data:
        _len_data.append(len(data))
    _len_data = np.array(_len_data)
    max_length = _len_data.max()

    padded_list_data = []
    mask = []
    
    for signal in list_data:
        signal = torch.tensor(signal, dtype=torch.float32)

        length = signal.size(0)
        pad_length = max_length - length
        
        # 신호 패딩
        padded_signal = torch.cat([signal, torch.full((pad_length,), padding_value)])
        padded_list_data.append(padded_signal)
        
        # 마스크 생성 (1: 유효 데이터, 0: 패딩된 부분)
        signal_mask = torch.cat([torch.ones(length), torch.zeros(pad_length)])
        mask.append(signal_mask)

    padded_list_data = torch.stack(padded_list_data).numpy()
    mask = torch.stack(mask).numpy()
    
    if verobose:
        print("Shape of data: {}".format(padded_list_data.shape))
        print("Shape of mask: {}".format(mask.shape))

    return padded_list_data, mask