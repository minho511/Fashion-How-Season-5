import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import timeit
import re
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import stats
from file_io import *
from requirement import *
from policy import *

import logging
from datetime import datetime

# https://gaussian37.github.io/dl-pytorch-lr_scheduler/
from scheduler import CosineAnnealingWarmUpRestarts

# # of items in fashion coordination
NUM_ITEM_IN_COORDI = 4
NUM_ITEM_IN_COORDI_ZSL = 5
#  # of metadata features    
NUM_META_FEAT = 4
# # of fashion coordination candidates        
NUM_RANKING = 3
# image feature size 
IMG_FEAT_SIZE = 2048
# augmentation ratio (aug2:aug1:org)
AUGMENTATION_RATIO = [0.5, 0.4, 0.1]


COSLABLE = [[1, 0, -1], [1, -1, 0], [-1, 1, 0], [0, -1, 1], [-1, 1, 0], [-1, 0, 1]]


class FahsionHowDataset(Dataset):
    """ Fashion-How dataset."""
    def __init__(self, in_file_trn_dialog, swer, mem_size, emb_size, 
                 crd_size, metadata, itm2idx, idx2itm, feats, num_rnk, 
                 similarities, corr_thres, eval_type, use_input_mask):
        """
        initialize your data, download, etc.
        """
        self._swer = swer
        self._mem_size = mem_size
        self._emb_size = emb_size
        self._crd_size = crd_size
        self._metadata = metadata
        self._itm2idx = itm2idx
        self._idx2itm = idx2itm
        self._feats = feats
        self._num_rnk = num_rnk
        self._similarities = similarities
        self._corr_thres = corr_thres
        self._eval_type = eval_type
        self._use_input_mask = use_input_mask
        self._datatype = ['aug2', 'aug1', 'org']
        self._dlg, self._crd = make_io_trn_data(
                        in_file_trn_dialog, self._itm2idx, self._idx2itm, 
                        self._similarities, self._num_rnk, self._corr_thres)
        self._len = len(self._dlg)
        self._num_item_in_coordi = len(self._crd[0][0])
    
    def __getitem__(self, index):
        """
        get item
        """
        datatype = np.random.choice(self._datatype, 1, p=AUGMENTATION_RATIO)
        if datatype == 'aug2':
            crd = []
            crd.append(self._crd[index][0])
            for j in range(1, self._num_rnk): 
                itm_lst = list(
                        permutations(np.arange(self._num_item_in_coordi), j)) 
                idx = np.arange(len(itm_lst))
                np.random.shuffle(idx)
                crd_new = replace_item(self._crd[index][0], self._itm2idx, 
                                       self._idx2itm, self._similarities, 
                                       itm_lst[idx[0]], self._corr_thres)
                crd.append(crd_new)
        elif datatype == 'aug1':
            crd = []
            for j in range(self._num_rnk - 1):
                crd.append(self._crd[index][j])
            idx = np.arange(self._num_item_in_coordi)
            np.random.shuffle(idx)
            crd_new = replace_item(crd[self._num_rnk-2], self._itm2idx, 
                                   self._idx2itm, self._similarities, 
                                   [idx[0]], self._corr_thres)
            crd.append(crd_new)
        else:
            crd =self._crd[index]
        # embedding    
        vec_dialog = vectorize_dlg(self._swer, self._dlg[index])
        # memorize for end-to-end memory network    
        mem_dialog = memorize_dlg(vec_dialog, self._mem_size, self._emb_size)
        # fashion item numbering    
        idx_coordi = indexing_coordi_dlg(crd, self._crd_size, self._itm2idx)
        # convert fashion item to metadata
        vec_coordi = convert_dlg_coordi_to_metadata(idx_coordi, 
                                        self._crd_size, self._metadata, 
                                        self._eval_type, self._feats, 
                                        self._use_input_mask)
        return mem_dialog, vec_coordi
        
    def __len__(self):
        """
        return data length
        """
        return self._len


class Model(nn.Module):
    """ Model for AI fashion coordinator """
    def __init__(self, mode, req_net_type, eval_net_type,
                 emb_size, key_size, mem_size, 
                 meta_size, hops, item_size, 
                 coordi_size, eval_node, num_rnk, 
                 use_batch_norm, use_dropout, zero_prob,
                 tf_dropout, tf_nhead,
                 tf_ff_dim, tf_num_layers, 
                 use_multimodal, img_feat_size):
        """
        initialize and declare variables
        """
        super().__init__()
        # class instance for requirement estimation
        self._requirement = RequirementNet4(emb_size)
        # class instance for ranking
        self._policy = PolicyNet4(eval_net_type, emb_size, key_size, 
                                 item_size, meta_size, coordi_size, 
                                 eval_node, num_rnk, use_batch_norm,
                                 use_dropout, zero_prob, 
                                 tf_dropout, tf_nhead,
                                 tf_ff_dim, tf_num_layers,
                                 use_multimodal, img_feat_size)

    def forward(self, dlg, crd):
        req = self._requirement(dlg)
        logits, mat = self._policy(req, crd)
        preds = torch.argmax(logits, 1)
        if self.training:
            return logits, preds, mat
        else:
            return logits, preds


class gAIa(object):
    """ Class for AI fashion coordinator """
    def __init__(self, args, device, name='gAIa'):
        """
        initialize
        """
        self._device = device
        self._batch_size = args.batch_size
        self._model_path = args.model_path
        self._model_file = args.model_file
        self._epochs = args.epochs
        self._max_grad_norm = args.max_grad_norm
        self._save_freq = args.save_freq
        self._num_eval = args.evaluation_iteration
        self._using_quantization = args.using_quantization
        use_dropout = args.use_dropout
        if args.mode == 'test' or args.mode == 'zsl':
            use_dropout = False
            
        if args.mode == 'train':
            # logger
            if not os.path.exists('logs'):
                os.makedirs('logs')

            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            if logger.hasHandlers():
                logger.handlers.clear()
            now = datetime.now()
            file_handler = logging.FileHandler(f"logs/{now.strftime('%Y-%m-%d %H:%M:%S')}.log")
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)
            self.logger = logger
        # class instance for subword embedding
        swer = SubWordEmbReaderUtil(args.subWordEmb_path)
        self._emb_size = swer.get_emb_size()
        meta_size = NUM_META_FEAT
        # ZSL
        coordi_size = NUM_ITEM_IN_COORDI
        if args.mode == 'zsl' or args.mode == 'test':
            coordi_size = NUM_ITEM_IN_COORDI_ZSL
        feats_size = IMG_FEAT_SIZE
        self._num_rnk = NUM_RANKING
        self._rnk_lst = np.array(list(permutations(np.arange(self._num_rnk), 
                                                   self._num_rnk)))
        
        # read metadata DB
        metadata, idx2item, item2idx, item_size, \
            similarities, feats = make_metadata(args.in_file_fashion, 
                                        swer, coordi_size,
                                        meta_size, args.use_multimodal, 
                                        args.in_dir_img_feats, feats_size)
        
        # prepare DB for training
        if args.mode == 'train':
            # dataloader
            dataset = FahsionHowDataset(args.in_file_trn_dialog, swer, 
                                        args.mem_size, self._emb_size,
                                        coordi_size, metadata, item2idx, 
                                        idx2item, feats, self._num_rnk,
                                        similarities, args.corr_thres,
                                        args.eval_net_type,
                                        args.use_input_mask)
            self._num_examples = len(dataset)
            self._dataloader = DataLoader(dataset, 
                                          batch_size=self._batch_size, 
                                          shuffle=True)
        # prepare DB for evaluation
        elif args.mode == 'test' or args.mode == 'zsl':
            self._tst_dlg, self._tst_crd = make_io_eval_data( 
                    args.in_file_tst_dialog, swer, args.mem_size,
                    coordi_size, item2idx, metadata, args.eval_net_type, feats)
            self._num_examples = len(self._tst_dlg)         

        # model
        self._model = Model(args.mode, args.req_net_type, args.eval_net_type,
                            self._emb_size, args.key_size, args.mem_size, 
                            meta_size, args.hops, item_size, coordi_size, 
                            args.eval_node, self._num_rnk, args.use_batch_norm, 
                            use_dropout, args.eval_zero_prob, args.tf_dropout, 
                            args.tf_nhead, args.tf_ff_dim, args.tf_num_layers, 
                            args.use_multimodal, feats_size)
        
        print('\n<model parameters>')        
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)

        if args.mode == 'train':
            self._optimizer = optim.SGD(self._model.parameters(), lr = 5e-6)
            self._scheduler = CosineAnnealingWarmUpRestarts(self._optimizer, T_0=30, T_mult=1, eta_max = 5e-4, T_up = 10, gamma = 0.5)

    def _get_loss(self, batch):
        """
        calculate loss
        """
        dlg, crd = batch
        crd_shuffle = []
        rnk_shuffle = []
        for c in crd:
            rnk_rnd, crd_rnd = shuffle_one_coordi_and_ranking(
                                    self._rnk_lst, c, self._num_rnk)
            crd_shuffle.append(torch.stack(crd_rnd))                     
            rnk_shuffle.append(torch.tensor(rnk_rnd))
        crd = torch.stack(crd_shuffle)
        rnk = torch.stack(rnk_shuffle)                         
        dlg = dlg.type(torch.float32)
        crd = crd.type(torch.float32)
        logits, _, mat = self._model(dlg, crd)
        loss = 0.0
        loss2 = 0.0
        
        for i in range(len(logits)):
            probs = nn.functional.softmax(logits[i], dim=0)
            ## minho
            coslabel = torch.FloatTensor(COSLABLE[rnk[i]]).cuda()
            loss2 += nn.functional.mse_loss(mat[i], coslabel)
            for j in range(len(self._rnk_lst)):
                corr, _ = stats.weightedtau(
                                self._num_rnk-1-self._rnk_lst[rnk[i]], 
                                self._num_rnk-1-self._rnk_lst[j])
                loss += ((1.0 - corr) * 0.5 * probs[j])
        tot_loss = loss + loss2
        # tot_loss = loss
        return tot_loss

    def train(self):
        """
        training
        """
        self.logger.info('\n<Train>')
        self.logger.info('total examples in dataset: {}'.format(self._num_examples))
        if not os.path.exists(self._model_path):
            os.makedirs(self._model_path)
        init_epoch = 1        
        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                self._model.load_state_dict(checkpoint['model'])
                self.logger.info('[*] load success: {}\n'.format(file_name))
                init_epoch += int(re.findall('\d+', file_name)[-1])
            else:
                self.logger.info('[!] checkpoints path does not exist...\n')
                return False
        self._model.to(self._device)
        end_epoch = self._epochs + init_epoch
        for curr_epoch in range(init_epoch, end_epoch):
            time_start = timeit.default_timer()
            losses = []
            iter_bar = tqdm(self._dataloader)
            for batch in iter_bar:
                self._optimizer.zero_grad()
                batch = [t.to(self._device) for t in batch]
                loss = self._get_loss(batch).mean()
                loss.backward()
                # nn.utils.clip_grad_norm_(self._model.parameters(), 
                #                         self._max_grad_norm)
                self._optimizer.step()
                losses.append(loss)

            #minho
            self._scheduler.step()
            for i, param_group in enumerate(self._optimizer.param_groups):
                self.logger.info(f"Epoch {curr_epoch} : group {i} lr {param_group['lr']}")

            time_end = timeit.default_timer()
            self.logger.info('-'*30)
            self.logger.info('Epoch: {}/{}'.format(curr_epoch, end_epoch - 1))
            self.logger.info('Time: {:.2f}sec'.format(time_end - time_start))
            self.logger.info('Loss: {:.4f}'.format(torch.mean(torch.tensor(losses))))
            self.logger.info('-'*30)
            if curr_epoch % self._save_freq == 0:
                file_name = os.path.join(self._model_path, 
                                         'gAIa-{}.pt'.format(curr_epoch))
                if not self._using_quantization:
                    torch.save({'model': self._model.state_dict()}, file_name)
                else:
                    from torch.quantization import quantize_dynamic
                    self._model.to('cpu')
                    _model_quant = quantize_dynamic(self._model, {nn.Linear,nn.RNN}, dtype=torch.qint8)
                    torch.save({'model': _model_quant.state_dict()}, file_name)
                    self._model.to(self._device)

        self.logger.info('Done training; epoch limit {} reached.\n'.format(self._epochs))
        return True
        
    def _calculate_weighted_kendal_tau(self, pred, label):
        """
        calcuate Weighted Kendal Tau Correlation
        """
        total_count = 0
        total_corr = 0
        for p, l in zip(pred, label):
            corr, _ = stats.weightedtau(
                            self._num_rnk-1-self._rnk_lst[l], 
                            self._num_rnk-1-self._rnk_lst[p])
            total_corr += corr
            total_count += 1
        return (total_corr / total_count)
    
    def _predict(self, eval_dlg, eval_crd):
        """
        predict
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        eval_crd = torch.tensor(eval_crd).to(self._device)
        preds = []
        for start in range(0, eval_num_examples, self._batch_size):
            end = start + self._batch_size
            if end > eval_num_examples:
                end = eval_num_examples
            _, pred = self._model(eval_dlg[start:end],
                                  eval_crd[start:end])

            pred = pred.cpu().numpy()
            for j in range(end-start):
                preds.append(pred[j])
        preds = np.array(preds)
        return preds, eval_num_examples    
    
    def _evaluate(self, eval_dlg, eval_crd):
        """
        evaluate
        """
        eval_num_examples = eval_dlg.shape[0]
        eval_corr = []
        eval_dlg = torch.tensor(eval_dlg).to(self._device)
        for i in range(self._num_eval):
            preds = []
            # DB shuffling
            coordi, rnk = shuffle_coordi_and_ranking(eval_crd, self._num_rnk)
            coordi = torch.tensor(coordi).to(self._device)
            for start in range(0, eval_num_examples, self._batch_size):
                end = start + self._batch_size
                if end > eval_num_examples:
                    end = eval_num_examples
                _, pred = self._model(eval_dlg[start:end], 
                                      coordi[start:end])
                pred = pred.cpu().numpy()
                for j in range(end-start):
                    preds.append(pred[j])
            preds = np.array(preds)
            # compute Weighted Kendal Tau Correlation
            corr = self._calculate_weighted_kendal_tau(preds, rnk)
            eval_corr.append(corr)
        return np.array(eval_corr), eval_num_examples
    
    def zsl(self):
        """
        create prediction
        """
        print('\n<Predict>')

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                if self._using_quantization:
                    from torch.quantization import quantize_dynamic
                    self._device = 'cpu'
                    checkpoint = torch.load(file_name)
                    _model_quant = quantize_dynamic(self._model, {nn.Linear, nn.RNN}, dtype=torch.qint8).cpu()
                    _model_quant.load_state_dict(checkpoint['model'])
                    self._model = _model_quant
                    print('[*] load success: quantized {}\n'.format(file_name))
                else:
                    checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                    self._model.load_state_dict(checkpoint['model'])
                    self._model.to(self._device)
                    print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # predict
        preds, num_examples = self._predict(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*50)
        print('Prediction Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('-'*50) 
        return preds.astype(int)
        
    def test(self):
        """
        test        
        """
        print('\n<Test>')

        if self._model_file is not None:
            file_name = os.path.join(self._model_path, self._model_file)
            if os.path.exists(file_name):
                if self._using_quantization:
                    from torch.quantization import quantize_dynamic
                    self._device = 'cpu'
                    checkpoint = torch.load(file_name)
                    _model_quant = quantize_dynamic(self._model, {nn.Linear, nn.RNN}, dtype=torch.qint8).cpu()
                    _model_quant.load_state_dict(checkpoint['model'])
                    self._model = _model_quant
                    print('[*] load success: quantized {}\n'.format(file_name))
                else:
                    checkpoint = torch.load(file_name, map_location=torch.device(self._device))
                    self._model.load_state_dict(checkpoint['model'])
                    self._model.to(self._device)
                    print('[*] load success: {}\n'.format(file_name))
            else:
                print('[!] checkpoints path does not exist...\n')
                return False
        else:
            return False
        time_start = timeit.default_timer()
        # evluation
        self._model.eval()
        test_corr, num_examples = self._evaluate(self._tst_dlg, self._tst_crd)
        time_end = timeit.default_timer()
        print('-'*30)
        print('Test Time: {:.2f}sec'.format(time_end-time_start))
        print('# of Test Examples: {}'.format(num_examples))
        print('Test WKTC: {:.4f}'.format(np.mean(test_corr)))
        print('-'*30)
        return np.mean(test_corr)
