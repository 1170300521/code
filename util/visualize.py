import json
import fire
import os.path as osp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
import pandas as pd
import ast
from PIL import Image, ImageDraw, ImageFont
import tqdm

def save_visualize(outputs, visualize_dir):
    """
    Save self-attention and cross-modal attention weights to files along with
    imgs and word queries

    outputs: model outputs dict
    visualize_dir: path to save files
    """
    if len(outputs['ids']) == 0:
        return
    filename = osp.join(visualize_dir, str(outputs['ids'][0])+'.json')
    query_ids = outputs['pred_logits'][:,:,0]
    _, query_ids = query_ids.max(1)    
    att_dict = {
        'sents': outputs['sents'],
        'size': outputs['size'],
        'img': outputs['img'],
        'self_att': outputs['self_att'],
        'cross_att': outputs['cross_att'],
        'query_ids': query_ids.cpu().detach().tolist(),
    }
    with open(filename, 'w') as f:
        json.dump(att_dict, f)


def show(filename, split_sent=9):
    """
    Show self-attention and cross-attention
    Parameters:
        filename: filename of attention maps
        split_sent: the number of first N words to show
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    file_prefix = filename.split(".")[0]
    # get sentence words
    sents = [s.split() for s in data['sents']]
    self_att = np.array(data['self_att'])  # B x L x N x N
    cross_att = np.array(data['cross_att'])  # B x L x N x (H*W)
    b, l, n, _ = cross_att.shape
    # better visualization with seaborn
    cross_att = cross_att.reshape((b, l, n, 20, 20))
    subject = [(i,s[i] if i<len(s) else "PD") for s, i in zip(sents, data['query_ids'])]
    imgs = np.array(data['img']).transpose((0, 2, 3, 1))

    # self-attention visualization
#    for i in range(b):
#        title = "Idx: " + str(subject[i][0]) + "; Target: "+ subject[i][1] + "; " + data['sents'][i]
#        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[25, 15])
#        fig.suptitle(title, fontsize=30)
#        for j in range(l):
#            sns.heatmap(self_att[i][j][:split_sent, :split_sent], ax=axes[int(j/3)][j%3], 
#                        annot=self_att[i][j][:split_sent, :split_sent],cmap='YlGn')
#            axes[int(j/3)][j%3].set_title("Level " + str(j))
#        fig.savefig(file_prefix+"_selfattn_"+str(i)+".png")
#        plt.close(fig)

    # cross-attention visualization
    for i in range(b):
        title = "Idx: " + str(subject[i][0]) + "; Target: "+ subject[i][1] + "; " + data['sents'][i]
        fig, axes = plt.subplots(nrows=6, ncols=split_sent, figsize=[30, 18])
        fig.suptitle(title, fontsize=30)
        for j in range(l):
            for k in range(split_sent):
                sns.heatmap(cross_att[i][j][k], ax=axes[j][k], cmap='YlGn')
                word = sents[i][k] if k < len(sents[i]) else "PD"
                axes[j][k].set_title("Level {}:".format(j)+"; "+word)
        fig.savefig(file_prefix+"_crossattn_"+str(i)+".png")
        plt.close(fig)

    # images visualization
    for i in range(b):
        title = "Idx: " + str(subject[i][0]) + "; Target: "+ subject[i][1] + "; " + data['sents'][i]
        fig = plt.figure(i)
        fig.suptitle(title, fontsize=20)
        plt.imshow(imgs[i])
        fig.savefig(file_prefix+"_img_"+str(i)+".png")
        plt.close(fig)
    print("Complete show " + filename)


def read_annotations(ds_name, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_data['bbox'] = trn_data.bbox.apply(
            lambda x: ast.literal_eval(x))
        sample = trn_data['query'].iloc[0]
        if sample[0] == '[':
            trn_data['query'] = trn_data['query'].apply(
                lambda x: ast.literal_eval(x))

        trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
        trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1])
        trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
        trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])
        if ds_name == 'flickr30k':
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: f'{x}.jpg'))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif ds_name == 'refclef':
            trn_df = trn_data[['img_id',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif 'flickr30k_c' in ds_name:
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: x))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif ds_name in ['refcoco', 'refcoco+', 'refcocog']:
            trn_df = trn_data[['img_id',
                                'x1', 'y1', 'x2', 'y2', 'query']]
        else :
            raise RuntimeError("No dataset named {}".format(ds_name))
        return trn_df

def visual_dataset(ds_name, ds_dict):
    img_path = ds_dict[ds_name]['img_dir']
    for data_type in ds_dict[ds_name].keys():
        if data_type.startswith("val") or data_type.startswith("test"):
            gt_file = ds_dict[ds_name][data_type]
            gt_data = read_annotations(ds_name, gt_file)
            suffix = osp.basename(gt_file).split(".")[0]
            vis_img_path = osp.join(ds_dict[ds_name]['data_dir'], "vis_" +suffix)
            os.makedirs(vis_img_path, exist_ok=True)
            for idx in tqdm.tqdm(range(len(gt_data))):
                img_file, x1, y1, x2, y2, query = gt_data.iloc[idx]
                img = Image.open(osp.join(img_path, img_file))
                img = img.convert("RGB")
                new_img = ImageDraw.Draw(img)
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                new_img.rectangle(((x1, y1), (x2, y2)), outline=(0, 255, 0), width=3)
                new_img.text((0, 0), query, (255, 0, 255), font=ImageFont.truetype(font='./data/times.ttf', size=20))
                img.save(osp.join(vis_img_path, osp.basename(img_file)))


def visual_all_dataset():
    ds_lists = ['refcoco', 'refcoco+', 'refcocog', 'refclef', 'flickr30k']
    with open("./data/ds_info.json", 'r') as f:
        ds_dict = json.load(f)
    for ds_name in ds_lists:
        print("Dataset: " + ds_name)
        visual_dataset(ds_name, ds_dict)

if __name__ == "__main__":
    visual_all_dataset()
