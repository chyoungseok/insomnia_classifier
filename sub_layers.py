from torch import nn, Tensor
from torch.nn import functional as F

class ConvBnRelu(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 kernel: int,
                 stride: int,
                 is_max_pool: bool,
                 dilation: int = 1,
                 padding: str = " no",
                 activation: str = 'relu') -> None:
        super().__init__()
        self.conv = nn.Conv1d(ch_in, ch_out, kernel, stride, 
                              padding=0, bias=False)
        self.bn = nn.BatchNorm1d(ch_out)

        self.activation = None        
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.kernel = kernel
        self.stride = stride        
        self.dilation = dilation
        self.is_max_pool = is_max_pool
        self.padding = padding
        
    def forward(self, feat_in: Tensor) -> Tensor:
        if self.padding == 'same':
            feat_in = self.pad_for_same_size(feat_in, self.kernel, self.stride, self.dilation)

        f_map = self.conv(feat_in)
        f_map = self.bn(f_map)
        if self.activation is not None:
            f_map = self.activation(f_map)

        if self.is_max_pool:
            f_map = self.max_pool(f_map)

        return f_map
    
    @staticmethod
    def pad_for_same_size(x, kernel_size, stride, dilation):
        input_size = x.size(-1)

        total_padding = max(
            0,
            (input_size - 1) * stride + dilation * (kernel_size - 1) + 1 - input_size
        )
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        # 비대칭 패딩 적용
        x_padded = F.pad(x, (left_padding, right_padding))
        return x_padded