from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F
import torchvision.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import re
import PIL
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import pickle
import ast
from torchvision import transforms
import spacy
import math
import os.path as osp
import random
from util.data_utils import generate_iou_groundtruth, pad_object_maps, visual_sample
from util.misc import nested_tensor_from_tensor_list, tlbr2cthw
# from extended_config import cfg as conf


nlp = spacy.load('en_core_web_lg')

class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class VGDataset(Dataset):
    """
    Any Grounding dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, bbox, queries
        Can have same img_file on multiple lines
    """

    def __init__(self, cfg, json_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = json_file
        self.ds_name = ds_name
        self.split_type = split_type
        self.is_train = (self.split_type == 'train')
        self.use_mlm = cfg.use_mlm
        self.use_obj_att = not cfg.no_obj_att
        # self.image_data = pd.read_csv(csv_file)
        self.image_data = self._read_annotations(json_file)
        # self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])
        # self.phrase_len = cfg.phrase_len
        self.phrase_len = cfg.num_queries  # keep the same as detr
        self.item_getter = getattr(self, 'simple_item_getter')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)
    
    def get_bboxs(self, qlen, annot, img_h, img_w):
        bboxs = [[0.5, 0.5, 0.5, 0.5] for i in range(qlen)]
        bboxs = np.zeros((qlen, 4)) + 0.5
        labels = [1] * qlen
        for obj_annot in annot['objects']:
            if obj_annot['idx'] >= qlen:
                continue
            x1 = obj_annot['x']
            y1 = obj_annot['y']
            x2 = x1 + obj_annot['w']
            y2 = y1 + obj_annot['h']
            x1 = abs(x1 / img_w)
            x2 = abs(x2 / img_w)
            y1 = abs(y1 / img_h)
            y2 = abs(y2 / img_h)
            bboxs[obj_annot['idx']] = np.array([(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)])
            labels[obj_annot['idx']] = 0
        return bboxs, labels
    
    def get_object_maps(self, qlen, annot, img_h, img_w):
        img_h = math.ceil(img_h / 32)
        img_w = math.ceil(img_w / 32)
        att_maps = np.zeros([qlen, img_h, img_w])
        for obj_annot in annot['objects']:
            if obj_annot['idx'] >= qlen:
                continue
            x = obj_annot['x']
            y = obj_annot['y']
            w = obj_annot['w']
            h = obj_annot['h']
            x = int(x/32)
            y = int(y/32)
            h = math.ceil(h/32)
            w = math.ceil(w/32)
            word_map = generate_iou_groundtruth((img_h, img_w), (x, y), (h, w))
            word_map = np.clip(word_map, a_min=0, a_max=1)
            att_maps[obj_annot['idx'], :, :] = word_map
        return att_maps

    def get_attr_labels(self, qlen, annot):
        attr_labels = np.zeros((len(annot['attributes']), 66000))
        # print(annot)
        attr_ids = []
        for i, a in enumerate(annot['attributes']):
            if a['sent_idx'] >= qlen:
                continue
            attr_labels[i][a['attr_ids']] = 1
            attr_ids.append(a['sent_idx'])
        return attr_labels[:len(attr_ids)], attr_ids
    
    def get_rel_ids(self, qlen, annot):
        obj_ids = []
        sub_ids = []
        rel_ids = []
        for r in annot['relationships']:
            if r['obj_idx'] >= qlen or r['sub_idx'] >= qlen:
                continue
            obj_ids.append(r['obj_idx'])
            sub_ids.append(r['sub_idx'])
            rel_ids.append(r['rel_idx'])
        return obj_ids, sub_ids, rel_ids

    def simple_item_getter(self, idx):
        img_file, annot, q_chosen = self.load_annotations(idx)
        img = PIL.Image.open(img_file).convert('RGB')
        h, w = img.height, img.width
        
        # img_ = np.array(img)
        q_chosen = q_chosen.strip()
        sents = q_chosen
        q_chosen = 'ANS ' + q_chosen
        qtmp = nlp(str(q_chosen))

        qlen = min(len(qtmp), self.phrase_len)
        q_chosen_emb = qtmp[:qlen]
        bboxs, labels = self.get_bboxs(qlen, annot, h, w)
        if len(labels) == sum(labels):
            return self.simple_item_getter(idx + 1)
        if self.use_obj_att:
            obj_maps = self.get_object_maps(qlen, annot, h, w)
        q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
        # Add attributes
        attr_labels, attr_ids = self.get_attr_labels(qlen, annot)
        # qlen = len(q_chosen_emb_vecs)
        # Add relationships
        obj_ids, sub_ids, rel_ids = self.get_rel_ids(qlen, annot)
        img = self.transform(img)
        # visual_sample(img_file, bboxs, obj_maps, h, w, qtmp_words)
        out = {
            'img': img,
            'idxs': torch.tensor(idx).long(),
            'qvec': torch.from_numpy(q_chosen_emb_vecs).float(),
            'qlens': torch.tensor(qlen),
            'cthw': torch.tensor(bboxs).float(),
            'labels': torch.tensor(labels, dtype=torch.long).unsqueeze(-1),  # 0 reps object and 1 reps no-object
            'attr_labels': torch.tensor(attr_labels).float(),
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([h, w]),
            'sents': sents,
            'attr_ids': torch.tensor(attr_ids).long(),
            'obj_ids': torch.tensor(obj_ids).long(),
            'sub_ids': torch.tensor(sub_ids).long(),
            'rel_labels': torch.tensor(rel_ids).long(),
        }
        if self.use_obj_att:
            out['obj_maps'] = torch.tensor(obj_maps).float()
        
        return out

    def load_annotations(self, idx):
        img_file = str(self.image_data[idx]['image_id']) + '.jpg'
        img_file = osp.join(self.img_dir, img_file)
        sent = self.image_data[idx]['phrase']
        return img_file, self.image_data[idx], sent

    def _read_annotations(self, trn_file):
        with open(trn_file, 'r') as f:
            raw_data = json.load(f)
        data = []
        for i in raw_data:
            data.extend(i['regions'])
        return data


def collater(batch):
    # qlens = torch.Tensor([i['qlens'] for i in batch])
    # max_qlen = int(qlens.max().item())
    # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
    out_dict = {}
    for k in batch[0]:
        if k in ['sents', 'img', 'qvec', 'text_labels', 'masked_words', 'labels', \
            'obj_maps', 'cthw', 'attr_labels', 'attr_ids', 'obj_ids', 'sub_ids', 'rel_labels']:
            out_dict[k] = [b[k] for b in batch]
        else:
            out_dict[k] = torch.stack([b[k] for b in batch])
    if 'img' in batch[0].keys():
        out_dict['img'] = nested_tensor_from_tensor_list(out_dict['img'])
    if 'qvec' in batch[0].keys():
        out_dict['qvec'] = nested_tensor_from_tensor_list(out_dict['qvec'])
    if 'labels' in batch[0].keys():
        # batch * T * 1
        out_dict['labels'] = nested_tensor_from_tensor_list(out_dict['labels'])
    # if 'attr_labels' in batch[0].keys():
    #     out_dict['attr_labels'] = nested_tensor_from_tensor_list(out_dict['attr_labels'])
    if 'cthw' in batch[0].keys():
        # batch * T * 4
        out_dict['cthw'] = nested_tensor_from_tensor_list(out_dict['cthw'])
    if 'obj_maps' in batch[0].keys():
        max_len = out_dict['qvec'].mask.size(1)
        tmp_obj_maps = [pad_object_maps(obj_map, max_len) for obj_map in out_dict['obj_maps']]
        out_dict['obj_maps'] = nested_tensor_from_tensor_list(tmp_obj_maps)
    if 'text_labels' in batch[0].keys():
        max_len = max([len(l) for l in out_dict['text_labels']])
        text_labels_pad = torch.zeros(len(batch), max_len).long() - 1
        for i in range(len(batch)):
            labels_i = out_dict['text_labels'][i]
            text_labels_pad[i][0:len(labels_i)] = labels_i
        out_dict['text_labels'] = text_labels_pad
    if 'attr_ids' in batch[0].keys():
        out_dict['batch_attr'] = torch.cat([torch.full_like(attr, i) \
            for i, attr in enumerate(out_dict['attr_ids'])])
        out_dict['attr_ids'] = torch.cat(out_dict['attr_ids'])
        out_dict['attr_labels'] = torch.cat(out_dict['attr_labels'], dim=0)
    if 'sub_ids' in batch[0].keys():
        out_dict['batch_rel'] = torch.cat([torch.full_like(rel, i) \
            for i, rel in enumerate(out_dict['sub_ids'])])
        out_dict['sub_ids'] = torch.cat(out_dict['sub_ids'])
        out_dict['obj_ids'] = torch.cat(out_dict['obj_ids'])
        out_dict['rel_labels'] = torch.cat(out_dict['rel_labels'])

    return out_dict


def get_data(cfg, ds_info):
    # Get which dataset to use
    ds_name = cfg.ds_name
    trn_csv_file = ds_info[ds_name]['trn_csv_file']
    val_csv_file = ds_info[ds_name]['val_csv_file']
    if ds_name == 'pretrain':
        trn_ds = PretrainDataset(cfg=cfg, csv_file=trn_csv_file,
                        ds_name=ds_name, split_type='train', no_img=cfg.no_img)
        val_ds = PretrainDataset(cfg=cfg, csv_file=val_csv_file,
                          ds_name=ds_name, split_type='valid', no_img=cfg.no_img)
        test_ds = {'test': val_ds}
    else:
        trn_ds = VGDataset(cfg=cfg, json_file=trn_csv_file,
                            ds_name=ds_name, split_type='train')
        val_ds = VGDataset(cfg=cfg, json_file=val_csv_file,
                            ds_name=ds_name, split_type='valid')
        test_ds = {'test': val_ds}

    return {
        "train": trn_ds,
        "val": val_ds,
        "test": test_ds
    }


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


if __name__ == '__main__':
    data = get_data(cfg, ds_name='refclef')
