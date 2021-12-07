import torch
import torch.nn as nn
import torch.nn.quantized as qnn

torch.backends.quantized.engine = "qnnpack"


class model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 1)
        self.conv2 = nn.Conv1d(1, 1, 1)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.ff = qnn.FloatFunctional()

    def forward(self, x):
        x = self.quant(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y = self.ff.cat([x1, x2], 1)
        return self.dequant(y)


m = model().eval()
m.qconfig = torch.quantization.get_default_qconfig('qnnpack')

m = torch.quantization.prepare(m)
m(torch.randn(1, 1, 100))

m = torch.quantization.convert(m)

print(m(torch.randn(1, 1, 100)).shape)
