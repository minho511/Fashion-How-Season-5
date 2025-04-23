import torch
import torch.nn as nn
import torch.nn.functional as F

class Eval_Head(nn.Module):
    def __init__(self, out_c = 4):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.act = nn.LeakyReLU()
        self.fin = nn.Linear(3*3*out_c, 6, bias=False)
        self.layernorm = nn.LayerNorm(3*3*out_c)

    def forward(self, x):
        o1 = self.conv_layer(x)
        o2 = o1.flatten(1)
        o2 = self.layernorm(o2)
        o2 = self.act(o2)
        out_rnk = self.fin(o2)
        return out_rnk
    
class PolicyNet4(nn.Module):
    """Class for policy network"""
    def __init__(self, net_type, emb_size, key_size, item_size, meta_size, 
                 coordi_size, eval_node, num_rnk, use_batch_norm, 
                 use_dropout, eval_zero_prob, tf_dropout, tf_nhead, 
                 tf_ff_dim, tf_num_layers, use_multimodal,
                 img_feat_size, name='PolicyNet'):
        super().__init__()
        self._item_size = item_size
        self._emb_size = emb_size
        self._key_size = key_size
        self._meta_size = meta_size
        self._coordi_size = coordi_size
        self._num_rnk = num_rnk
        self._name = name
        self._net_type = net_type
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # minho
        self.out_c = 4

        # Transformer
        if self._net_type == 'tf':
            buf = eval_node[1:-1]
            buf = list(map(int, buf.split(',')))
            self._eval_out_node = buf[0]
            self._num_hid_rnk = buf[1:]
            self._num_hid_layer_rnk = len(self._num_hid_rnk)

            self._count_eval = 0
            if use_dropout:
                dropout = tf_dropout
            else:
                dropout = 0.0
                eval_zero_prob = 0.0
            num_in = self._emb_size * self._meta_size 
            if use_multimodal:
                num_in += img_feat_size             
            self._summary1 = nn.Linear(512, self._key_size//2)
            self._summary2 = nn.Linear(512, self._key_size//2)
            self.eval_head = Eval_Head(out_c = self.out_c)

    def _evaluate_coordi(self, crd, req):
        """
        evaluate candidates
        """
        bs = crd.shape[0]
        enc_m = torch.permute(crd, (0, 2, 1))
        enc_m = self.avg_pool(enc_m).squeeze()
        enc_m = enc_m.view(bs, 5, 512)
        evl_rnk1 = self._summary1(enc_m[:, :4, :]).mean(1).squeeze()
        evl_rnk2 = self._summary2(enc_m[:, -1, :]).squeeze()
        evl_rnk = torch.cat([evl_rnk1, evl_rnk2], dim = -1)
        return evl_rnk


    def forward(self, req, crd):
        crd_tr = torch.transpose(crd, 1, 0)
        candi = []
        for i in range(self._num_rnk):
            crd_eval = self._evaluate_coordi(crd_tr[i], req)
            crd_eval = torch.nn.functional.normalize(crd_eval, dim = -1)
            candi.append(crd_eval.unsqueeze(1))
        candi.append(torch.nn.functional.normalize(req, dim = -1).unsqueeze(1))
        candi = torch.cat(candi, dim = 1) # 100x4x300
        result = torch.tril(torch.bmm(candi, candi.transpose(1, 2)), diagonal=-1)[:, 1:, :-1]
        result = result.unsqueeze(1)
        out_rnk = self.eval_head(result)
        return out_rnk, result[:, :, -1, :].squeeze()
        