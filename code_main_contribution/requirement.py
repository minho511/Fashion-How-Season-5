import torch
import torch.nn as nn
import torch.nn.functional as F

class RequirementNet4(nn.Module):
    """Requirement Network"""
    def __init__(self, emb_size, name='RequirementNet'):
        """
        initialize and declare variables
        """
        super().__init__()
        num_heads = 1
        num_layers = 1
        self.rnn = nn.RNN(emb_size, emb_size, num_layers=1)


    def forward(self, dlg):
        """
        build graph for requirement estimation
        """
        inputs = torch.transpose(dlg, 0, 1)
        enc = self.rnn(inputs)
        output = enc[0][-1]
        return output

if __name__ == '__main__':
    model = RequirementNet4(128)
    from torch.quantization import quantize_dynamic
    model.to('cpu')
    model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    torch.save({'model': model.state_dict()}, 'reqnet.pt')