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


import sys
import numpy as np
import csv
import codecs
import re
import os
import json
import _pickle as pickle
from ctypes import cdll, create_string_buffer
from itertools import permutations
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# For ZSL
# # of bags
NUM_FASION_ITEM_BG = 420
# # of scarves
NUM_FASION_ITEM_SC = 300
# # of hats
NUM_FASION_ITEM_HC = 356
# masking ratio (none:img:txt)
MASKING_RATIO = [0.4, 0.3, 0.3]


def _load_fashion_item(in_file, coordi_size, meta_size):
    """
    function: load fashion item metadata
    """
    print('loading fashion item metadata')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        names = []
        metadata = []
        prev_name = ''
        prev_feat = ''
        data = ''
        for l in fin.readlines():
            line = l.strip()
            w = line.split()
            name = w[0]
            if name != prev_name:
                names.append(name)
                prev_name = name
            feat = w[3]
            if feat != prev_feat:
                if prev_feat != '':
                    metadata.append(data)
                data = w[4]
                for d in w[5:]:
                    data += ' ' + d
                prev_feat = feat
            else:
                for d in w[4:]:
                    data += ' ' + d
        metadata.append(data)
        for i in range(coordi_size*meta_size):
            metadata.append('')
        # add null types    
        names.append('NONE-OUTER')
        names.append('NONE-TOP')
        names.append('NONE-BOTTOM')
        names.append('NONE-SHOES')
        # ZSL
        if coordi_size == 5:
            names.append('NONE-ACCESSARY')
            for n in range(NUM_FASION_ITEM_BG):
                for i in range(meta_size):
                    metadata.append('')
                fname = 'BG-' + str(n+1).zfill(3)    
                names.append(fname)
            for n in range(NUM_FASION_ITEM_SC):
                for i in range(meta_size):
                    metadata.append('')
                fname = 'SC-' + str(n+1).zfill(3)    
                names.append(fname)
            for n in range(NUM_FASION_ITEM_HC):
                for i in range(meta_size):
                    metadata.append('')
                fname = 'HC-' + str(n+1).zfill(3)    
                names.append(fname)
    return names, metadata


def _position_of_fashion_item(item):
    """
    function: get position of fashion items    
    """
    prefix = item[0:2]
    if prefix=='JK' or prefix=='JP' or prefix=='CT' or prefix=='CD' \
        or prefix=='VT' or item=='NONE-OUTER':
        idx = 0 
    elif prefix=='KN' or prefix=='SW' or prefix=='SH' or prefix=='BL' \
        or item=='NONE-TOP':
        idx = 1
    elif prefix=='SK' or prefix=='PT' or prefix=='OP' or item=='NONE-BOTTOM':
        idx = 2
    elif prefix=='SE' or item=='NONE-SHOES':
        idx = 3
    elif prefix=='BG' or prefix=='SC' or prefix=='HC' or item=='NONE-ACCESSARY':
        idx = 4
    else:
        raise ValueError('{} do not exists.'.format(item))
    return idx


def _insert_into_fashion_coordi(coordi, items):
    """
    function: insert new items into previous fashion coordination
    """
    new_coordi = coordi[:]
    for item in items:
        item = item.split(';')
        new_item = item[len(item)-1].split('_')
        cl_new_item = new_item[len(new_item)-1]
        pos = _position_of_fashion_item(cl_new_item)
        if cl_new_item[0:2]=='OP':
            new_coordi[1] = 'NONE-TOP'
        new_coordi[pos] = cl_new_item
    return new_coordi


def _load_trn_dialog(in_file):
    """
    function: load training dialog DB    
    """
    print('loading dialog DB')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        data_utter = []
        data_coordi = []
        data_reward = []
        delim_utter = []
        delim_coordi = []
        delim_reward = []
        num_dialog = 0    
        num_turn = 1
        num_coordi = 0
        num_reward = 0
        is_first = True
        for l in fin.readlines():
            line = l.strip()
            w = line.split()
            ID = w[1]
            if w[0] == '0':
                if is_first:
                    is_first = False
                else:
                    data_utter.append(tot_utter.strip())                  
                    if prev_ID == '<CO>':
                        data_coordi.append(coordi)
                        num_coordi += 1
                    if prev_ID == '<US>':
                        data_reward.append(tot_func.strip())
                        num_reward += 1
                    delim_utter.append(num_turn)
                    delim_coordi.append(num_coordi)
                    delim_reward.append(num_reward)
                    num_turn += 1
                prev_ID = ID
                tot_utter = ''
                tot_func = ''
                coordi = ['NONE-OUTER', 
                          'NONE-TOP',  
                          'NONE-BOTTOM', 
                          'NONE-SHOES']
                num_dialog += 1
            if ID == '<AC>':
                items = w[2:]
                coordi = _insert_into_fashion_coordi(coordi, items)
                utter = ''
                continue
            func = re.sub(pattern='[^A-Z_;]', repl='', string=w[-1])
            func = re.sub(pattern='[;]', repl=' ', string=func)
            if func == '_':
                func = ''
            if func != '':
                w = w[:-1]    
            if prev_ID != ID:
                data_utter.append(tot_utter.strip())                  
                if prev_ID == '<CO>':
                    data_coordi.append(coordi)
                    num_coordi += 1
                if prev_ID == '<US>':
                    data_reward.append(tot_func.strip())
                    num_reward += 1
                tot_utter = ''
                tot_func = ''
                prev_ID = ID
                num_turn += 1
            for u in w[2:]:
                tot_utter += ' ' + u 
            tot_func += ' ' + func
        data_utter.append(tot_utter.strip())                  
        delim_utter.append(num_turn)
        if prev_ID == '<CO>':
            data_coordi.append(coordi)
            num_coordi += 1
        if prev_ID == '<US>':
            data_reward.append(tot_func.strip())
            num_reward += 1
        delim_coordi.append(num_coordi)
        delim_reward.append(num_reward)
        print('# of dialog: {} sets'.format(num_dialog))
        # only use last reward
        data_reward_last = []
        for r in data_reward:
            r = r.split()
            if len(r) >= 1:
                data_reward_last.append(r[len(r)-1])    
            else:
                data_reward_last.append('')
        return data_utter, data_coordi, data_reward_last, \
               np.array(delim_utter, dtype='int32'), \
               np.array(delim_coordi, dtype='int32'), \
               np.array(delim_reward, dtype='int32')


def _load_eval_dialog(in_file, coordi_size):
    """
    function: load test dialog DB    
    """
    print('loading dialog DB')
    with open(in_file, encoding='euc-kr', mode='r') as fin:
        data_utter = []
        data_coordi = []
        num_dialog = 0
        num_utter = 0
        is_first = True
        for line in fin.readlines():
            line = line.strip()
            if line[0] == ';':
                if line[2:5] == 'end':
                    break
                if is_first:
                    is_first = False
                else:
                    data_utter.append(tot_utter)
                    data_coordi.append(tot_coordi)
                tot_utter = []
                tot_coordi = []
                num_dialog += 1
            elif line[0:2] == 'US' or line[0:2] == 'CO':
                utter = line[2:].strip()
                tot_utter.append(utter)
                num_utter += 1
            elif line[0] == 'R':
                coordi = line[2:].strip()
                if coordi_size == 4:
                    new_coordi = ['NONE-OUTER', 
                                  'NONE-TOP',  
                                  'NONE-BOTTOM', 
                                  'NONE-SHOES']
                elif coordi_size == 5:
                    new_coordi = ['NONE-OUTER', 
                                  'NONE-TOP',  
                                  'NONE-BOTTOM', 
                                  'NONE-SHOES',
                                  'NONE-ACCESSARY']
                new_coordi = _insert_into_fashion_coordi(new_coordi, 
                                                         coordi.split())
                tot_coordi.append(new_coordi)
        if not is_first:
            data_utter.append(tot_utter)
            data_coordi.append(tot_coordi)
        print('# of dialog: {} sets'.format(num_dialog))
        return data_utter, data_coordi
        

class SubWordEmbReaderUtil:
    """
    Class for subword embedding    
    """
    def __init__(self, data_path):
        """
        initialize    
        """
        print('\n<Initialize subword embedding>')
        print ('loading=', data_path)
        with open(data_path, 'rb') as fp:
            self._subw_length_min = pickle.load(fp)
            self._subw_length_max = pickle.load(fp)
            self._subw_dic = pickle.load(fp, encoding='euc-kr')
            self._emb_np = pickle.load(fp, encoding='bytes')
            self._emb_size = self._emb_np.shape[1]

    def get_emb_size(self):
        """
        get embedding size    
        """
        return self._emb_size        

    def _normalize_func(self, s):
        """
        normalize
        """
        s1 = re.sub(' ', '', s)
        s1 = re.sub('\n', 'e', s1)
        sl = list(s1)
        for a in range(len(sl)):
            if sl[a].encode('euc-kr') >= b'\xca\xa1' and \
               sl[a].encode('euc-kr') <= b'\xfd\xfe': sl[a] = 'h'
        s1 = ''.join(sl)
        return s1

    def _word2syllables(self, word):
        """
        word to syllables
        """
        syl_list = []

        dec = codecs.lookup('cp949').incrementaldecoder()
        w = self._normalize_func(dec.decode(word.encode('euc-kr')))
        for a in list(w):
            syl_list.append(a.encode('euc-kr').decode('euc-kr'))
        return syl_list

    def _get_cngram_syllable_wo_dic(self, word, min, max):
        """
        get syllables
        """
        word = word.replace('_', '')
        p_syl_list = self._word2syllables(word.upper())  
        subword = []
        syl_list = p_syl_list[:]
        syl_list.insert(0, '<')
        syl_list.append('>')
        for a in range(len(syl_list)):
            for b in range(min, max+1):
                if a+b > len(syl_list): break
                x = syl_list[a:a+b]
                k = '_'.join(x)
                subword.append(k)
        return subword

    def _get_word_emb(self, w):
        """
        do word embedding
        """
        word = w.strip()
        assert len(word) > 0
        cng = self._get_cngram_syllable_wo_dic(word, self._subw_length_min, 
                                               self._subw_length_max)
        lswi = [self._subw_dic[subw] for subw in cng if subw in self._subw_dic]
        if lswi == []: lswi = [self._subw_dic['UNK_SUBWORD']]
        d = np.sum(np.take(self._emb_np, lswi, axis=0), axis = 0)
        return d

    def _get_sent_emb(self, s):
        """
        do sentence embedding
        """
        if s != '':
            s = s.strip().split()
            semb_tmp = []
            for a in s:
                semb_tmp.append(self._get_word_emb(a))
            avg = np.average(semb_tmp, axis=0)
        else:
            avg = np.zeros(self._emb_size)
        return avg


def _vectorize_sent(swer, sent):
    """
    function: vectorize one sentence    
    """
    vec_sent = swer._get_sent_emb(sent)
    return vec_sent 


def vectorize_dlg(swer, dialog):
    """
    function: vectorize one dialog    
    """
    vec_dlg = []
    for sent in dialog:
        sent_emb = _vectorize_sent(swer, sent)
        vec_dlg.append(sent_emb)
    vec_dlg = np.array(vec_dlg, dtype='float32')
    return vec_dlg


def _vectorize(swer, data):
    """
    function: vectorize dialogs    
    """
    print('vectorizing data')
    vec = []
    for dlg in data:
        dlg_emb = vectorize_dlg(swer, dlg)
        vec.append(dlg_emb)
    vec = np.array(vec, dtype=object)
    return vec
    

def memorize_dlg(dialog, mem_size, emb_size):
    """
    function: memorize one dialog for end-to-end memory network        
    """
    zero_emb = np.zeros((1, emb_size))
    idx = max(0, len(dialog) - mem_size)
    ss = dialog[idx:]
    pad = mem_size - len(ss)  
    for i in range(pad):
        ss = np.append(ss, zero_emb, axis=0)
    return np.array(ss, dtype='float32')
    

def _memorize(dialog, mem_size, emb_size):
    """
    function: memorize dialogs for end-to-end memory network        
    """
    print('memorizing data')
    memory = []
    for i in range(len(dialog)):
        ss = memorize_dlg(dialog[i], mem_size, emb_size)    
        memory.append(ss)
    return np.array(memory, dtype='float32')
    

def _make_ranking_examples(dialog, coordi, reward, item2idx, idx2item, 
                similarities, num_rank, corr_thres):
    """
    function: make candidates for training       
    """
    print('making ranking_examples')
    data_dialog = []
    data_coordi = []
    idx = np.arange(num_rank)
    num_item_in_coordi = len(coordi[0][0])
    for i in range(len(coordi)):
        crd_lst = coordi[i]
        crd_lst = crd_lst[::-1]
        crd = []
        prev_crd = ['', '', '', '']
        count = 0
        for j in range(len(crd_lst)):
            if crd_lst[j] != prev_crd and crd_lst[j] != \
                ['NONE-OUTER', 'NONE-TOP', 'NONE-BOTTOM', 'NONE-SHOES']:
                crd.append(crd_lst[j]) 
                prev_crd = crd_lst[j]
                count += 1
        rwd_lst = reward[i]
        rwd_lst = rwd_lst[::-1]
        rwd = ''
        for j in range(len(rwd_lst)):
            if rwd_lst[j] != '':
                rwd = rwd_lst[j]
                break
        if count >= num_rank:    
            for k in range(count - num_rank + 1):
                data_dialog.append(dialog[i])
                data_coordi.append(crd[k:(num_rank+k)])
            crd_aug = []
            crd_aug.append(crd[count-2])
            crd_aug.append(crd[count-1])
            idx = []
            for j in range(len(crd[0])):
                if not 'NONE' in crd[count-1][j]:
                    idx.append(j)
            np.random.shuffle(idx)
            crd_new = replace_item(crd[count-1], item2idx, idx2item, 
                                   similarities, idx, corr_thres)
            crd_aug.append(crd_new)
            data_dialog.append(dialog[i])
            data_coordi.append(crd_aug)
            crd_aug = []
            crd_aug.append(crd[count-1])
            for j in range(1, 3): 
                itm_lst = list(
                            permutations(np.arange(num_item_in_coordi), j)) 
                idx = np.arange(len(itm_lst))
                np.random.shuffle(idx)
                crd_new = replace_item(crd[count-1], item2idx, idx2item, 
                                similarities, itm_lst[idx[0]], corr_thres)
                crd_aug.append(crd_new)
            data_dialog.append(dialog[i])
            data_coordi.append(crd_aug)
        elif count == (num_rank - 1):
            crd_aug = []
            crd_aug.append(crd[count-2])
            crd_aug.append(crd[count-1])
            idx = []
            for j in range(len(crd[0])):
                if not 'NONE' in crd[count-1][j]:
                    idx.append(j)
            np.random.shuffle(idx)
            crd_new = replace_item(crd[count-1], item2idx, idx2item, 
                                   similarities, idx, corr_thres)
            crd_aug.append(crd_new)
            data_dialog.append(dialog[i])
            data_coordi.append(crd_aug)
            crd_aug = []
            crd_aug.append(crd[count-1])
            for j in range(1, 3): 
                itm_lst = list(
                            permutations(np.arange(num_item_in_coordi), j)) 
                idx = np.arange(len(itm_lst))
                np.random.shuffle(idx)
                crd_new = replace_item(crd[count-1], item2idx, idx2item, 
                                similarities, itm_lst[idx[0]], corr_thres)
                crd_aug.append(crd_new)
            data_dialog.append(dialog[i])
            data_coordi.append(crd_aug)
        elif count == (num_rank - 2):
            crd_aug = []
            crd_aug.append(crd[count-1])
            for j in range(1, 3): 
                itm_lst = list(
                            permutations(np.arange(num_item_in_coordi), j)) 
                idx = np.arange(len(itm_lst))
                np.random.shuffle(idx)
                crd_new = replace_item(crd[count-1], item2idx, idx2item, 
                                similarities, itm_lst[idx[0]], corr_thres)
                crd_aug.append(crd_new)
            data_dialog.append(dialog[i])
            data_coordi.append(crd_aug)
    return data_dialog, data_coordi
    

def replace_item(crd, item2idx, idx2item, similarities, pos, thres):
    """
    function: replace item using cosine similarities       
    """
    new_crd = crd[:]
    for p in pos:
        itm = crd[p]
        itm_idx = item2idx[p][itm]    
        idx = np.arange(len(item2idx[p]))
        np.random.shuffle(idx)
        for k in range(len(item2idx[p])):
            if similarities[p][itm_idx][idx[k]] < thres:
                rep_idx = idx[k]
                rep_itm = idx2item[p][rep_idx]
                break
        new_crd[p] = rep_itm
    return new_crd


def indexing_coordi_dlg(data, coordi_size, itm2idx):
    """
    function: fashion item numbering
    """
    vec_crd = []
    for itm in data:
        ss = np.array([itm2idx[j][itm[j]] for j in range(coordi_size)])
        vec_crd.append(ss)
    vec_crd = np.array(vec_crd, dtype='int32')
    return vec_crd


def _indexing_coordi(data, coordi_size, itm2idx):
    """
    function: fashion item numbering
    """
    print('indexing fashion coordi')
    vec = []
    for d in range(len(data)):
        vec_crd = indexing_coordi_dlg(data[d], coordi_size, itm2idx)
        vec.append(vec_crd)
    return np.array(vec, dtype='int32')


def _convert_one_coordi_to_metadata(one_coordi, coordi_size, 
                                    metadata, eval_type, 
                                    img_feats, mask_mode=False):
    """
    function: convert fashion coordination to metadata
    """
    mask_type_enum = ['mask_none', 'mask_img', 'mask_txt']
    if img_feats is None:
        items = []
        for j in range(coordi_size):
            buf = metadata[j][one_coordi[j]]
            items.append(buf)    
        items = np.stack(items, axis=0)      
    else:
        if eval_type == 'tf':
            items = []
            for j in range(coordi_size):
                mask_type = 'mask_none'
                if mask_mode:
                    mask_type = np.random.choice(mask_type_enum, 
                                                1, p=MASKING_RATIO)
                if mask_type == 'mask_none':
                    buf_meta = metadata[j][one_coordi[j]]
                    buf_feat = img_feats[j][one_coordi[j]]
                    buf = np.concatenate([buf_meta, buf_feat], axis=0)
                elif mask_type == 'mask_img':
                    buf_meta = metadata[j][one_coordi[j]]
                    buf_feat = np.zeros_like(img_feats[j][one_coordi[j]])
                    buf = np.concatenate([buf_meta, buf_feat], axis=0)
                elif mask_type == 'mask_txt':
                    buf_meta = np.zeros_like(metadata[j][one_coordi[j]])
                    buf_feat = img_feats[j][one_coordi[j]]
                    buf = np.concatenate([buf_meta, buf_feat], axis=0)
                items.append(buf)
            items = np.stack(items, axis=0) 
        elif eval_type == 'mlp':
            items_meta = None
            items_feat = None
            for j in range(coordi_size):
                mask_type = 'mask_none'
                if mask_mode:
                    mask_type = np.random.choice(mask_type_enum, 
                                                1, p=MASKING_RATIO)
                if mask_type == 'mask_none':
                    buf_meta = metadata[j][one_coordi[j]]
                    buf_feat = img_feats[j][one_coordi[j]]
                elif mask_type == 'mask_img':
                    buf_meta = metadata[j][one_coordi[j]]
                    buf_feat = np.zeros_like(img_feats[j][one_coordi[j]])
                elif mask_type == 'mask_txt':
                    buf_meta = np.zeros_like(metadata[j][one_coordi[j]])
                    buf_feat = img_feats[j][one_coordi[j]]
                if j == 0:
                    items_meta = buf_meta[:]
                    items_feat = buf_feat[:]
                else:
                    items_meta = np.concatenate(
                                    [items_meta[:], buf_meta[:]], axis=0)
                    items_feat += buf_feat[:]
            items_feat /= (float)(coordi_size)
            items = np.concatenate([items_meta, items_feat], axis=0)
    return items 
    
def convert_dlg_coordi_to_metadata(dlg_coordi, coordi_size, 
                metadata, eval_type, img_feats, mask_mode=False):
    """
    function: convert fashion coordinations to metadata
    """
    items = _convert_one_coordi_to_metadata(dlg_coordi[0], 
                coordi_size, metadata, eval_type, img_feats, mask_mode)
    prev_coordi = dlg_coordi[0][:]
    prev_items = items[:]
    scripts = np.expand_dims(items, axis=0)[:]
    for i in range(1, dlg_coordi.shape[0]):
        if np.array_equal(prev_coordi, dlg_coordi[i]):
            items = prev_items[:] 
        else:
            items = _convert_one_coordi_to_metadata(dlg_coordi[i], 
                                    coordi_size, metadata, eval_type, 
                                    img_feats, mask_mode)
        prev_coordi = dlg_coordi[i][:]
        prev_items = items[:]
        items = np.expand_dims(items, axis=0)
        scripts = np.concatenate([scripts[:], items[:]], axis=0)
    return scripts


def _convert_coordi_to_metadata(coordi, coordi_size, metadata, 
                        eval_type, img_feats, mask_mode=False):
    """
    function: convert fashion coordinations to metadata
    """
    print('converting fashion coordi to metadata')
    vec = []
    for d in range(len(coordi)):
        vec_meta = convert_dlg_coordi_to_metadata(coordi[d], 
                                coordi_size, metadata, 
                                eval_type, img_feats, mask_mode)
        vec.append(vec_meta)
    return np.array(vec, dtype='float32')


def _episode_slice(data, delim):
    """
    function: divide by episode
    """
    episodes = []
    start = 0
    for end in delim:
        epi = data[start:end]
        episodes.append(epi)
        start = end
    return episodes


def _categorize(name, vec_item, coordi_size):
    """
    function: categorize fashion items    
    """
    slot_item = [] 
    slot_name = []
    for i in range(coordi_size):
        slot_item.append([])
        slot_name.append([])
    for i in range(len(name)):
        pos = _position_of_fashion_item(name[i])
        slot_item[pos].append(vec_item[i])
        slot_name[pos].append(name[i])
    slot_item = np.array([np.array(s) for s in slot_item],
                         dtype=object)
    return slot_name, slot_item


def shuffle_one_coordi_and_ranking(rank_lst, coordi, num_rank):
    """
    function: shuffle fashion coordinations   
    """
    idx = np.arange(num_rank)
    np.random.shuffle(idx)
    for k in range(len(rank_lst)):
        if np.array_equal(idx, rank_lst[k]):
            rank = k
            break
    rand_crd = []
    for k in range(num_rank):
        rand_crd.append(coordi[idx[k]])
    return rank, rand_crd


def shuffle_coordi_and_ranking(coordi, num_rank):
    """
    function: shuffle fashion coordinations   
    """
    data_rank = []
    data_coordi_rand = []        
    idx = np.arange(num_rank)
    rank_lst = np.array(list(permutations(idx, num_rank)))
    for i in range(len(coordi)):
        idx = np.arange(num_rank)
        np.random.shuffle(idx)
        for k in range(len(rank_lst)):
            if np.array_equal(idx, rank_lst[k]):
                rank = k
                break
        data_rank.append(rank)
        coordi_rand = []
        crd = coordi[i]
        for k in range(num_rank):
            coordi_rand.append(crd[idx[k]])
        data_coordi_rand.append(coordi_rand)
    data_coordi_rand = np.array(data_coordi_rand, dtype='float32')    
    data_rank = np.array(data_rank, dtype='int32')
    return data_coordi_rand, data_rank


def _load_fashion_feature(dir_name, slot_name, coordi_size, feat_size):
    """
    function: load image features
    """
    suffix = '.npy'
    feats = []
    for i in range(coordi_size):
        feat = []    
        for n in slot_name[i]:
            if n[0:4] == 'NONE':
                feat.append(np.zeros((feat_size)))
            else:
                img_name = dir_name + '/' + n + suffix
                with open(img_name, 'r') as fin:
                    data = np.load(img_name)    
                    feat.append(np.mean(data, axis=0))
        feats.append(np.array(feat))
    feats = np.array(feats, dtype=object)            
    return feats
    

def make_metadata(in_file_fashion, swer, coordi_size, meta_size,
                  use_multimodal, in_dir_img_feats, feat_size):
    """
    function: make metadata for training and test
    """
    print('\n<Make metadata>')
    if not os.path.exists(in_file_fashion):
        raise ValueError('{} do not exists.'.format(in_file_fashion))
    # load metadata DB    
    name, data_item = _load_fashion_item(in_file_fashion, coordi_size,
                                         meta_size)
    print('vectorizing data')
    emb_size = swer.get_emb_size()
    # embedding    
    vec_item = vectorize_dlg(swer, data_item)
    vec_item = vec_item.reshape((-1, meta_size*emb_size))
    # categorize fashion items    
    slot_name, slot_item = _categorize(name, vec_item, coordi_size)
    slot_feat = None
    if use_multimodal:
        slot_feat = _load_fashion_feature(in_dir_img_feats, 
                                    slot_name, coordi_size, feat_size)
    vec_similarities = []
    # calculation cosine similarities
    for i in range(coordi_size):
        item_sparse = sparse.csr_matrix(slot_item[i])
        similarities = cosine_similarity(item_sparse)
        vec_similarities.append(similarities)
    vec_similarities = np.array(vec_similarities, dtype=object)
    idx2item = []
    item2idx = []
    item_size = []
    for i in range(coordi_size):
        idx2item.append(dict((j, m) for j, m in enumerate(slot_name[i])))
        item2idx.append(dict((m, j) for j, m in enumerate(slot_name[i])))
        item_size.append(len(slot_name[i]))
    return slot_item, idx2item, item2idx, item_size, \
           vec_similarities, slot_feat


def make_io_trn_data(in_file_dialog, item2idx, idx2item, similarities, 
                     num_rank, corr_thres=1.0):
    """
    function: prepare DB for training
    """
    print('\n<Make input & output data>')
    if not os.path.exists(in_file_dialog):
        raise ValueError('{} do not exists.'.format(in_file_dialog))
    # load training dialog DB    
    dialog, coordi, reward, delim_dlg, delim_crd, delim_rwd = \
                                            _load_trn_dialog(in_file_dialog)
    # per episode
    dialog = _episode_slice(dialog, delim_dlg)
    coordi = _episode_slice(coordi, delim_crd)
    reward = _episode_slice(reward, delim_rwd)
    # prepare DB for evaluation
    data_dialog, data_coordi = \
                _make_ranking_examples(dialog, coordi, reward, item2idx, 
                                        idx2item, similarities, num_rank, 
                                        corr_thres)
    return data_dialog, data_coordi
    

def make_io_eval_data(in_file_dialog, swer, mem_size, coordi_size, 
                      item2idx, metadata, eval_type, img_feats=None):
    """
    function: prepare DB for test
    """
    print('\n<Make input & output data>')
    if not os.path.exists(in_file_dialog):
        raise ValueError('{} do not exists.'.format(in_file_dialog))
    # load test dialog DB    
    data_dialog, data_coordi = _load_eval_dialog(in_file_dialog, 
                                                 coordi_size)
    # embedding    
    vec_dialog = _vectorize(swer, data_dialog)
    emb_size = swer.get_emb_size()
    # memorize for end-to-end memory network    
    mem_dialog = _memorize(vec_dialog, mem_size, emb_size)
    # fashion item numbering    
    idx_coordi = _indexing_coordi(data_coordi, coordi_size, item2idx)
    # convert fashion item to metadata
    vec_coordi = _convert_coordi_to_metadata(idx_coordi, coordi_size, 
                                    metadata, eval_type, img_feats)
    return mem_dialog, vec_coordi
