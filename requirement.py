'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2023, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2023.02.22.
'''

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