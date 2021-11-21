import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def generate_iou_groundtruth(grid_shapes,true_xy,true_hw):
    """
    :param grid_shapes:   widths and heights for generation (h, w)
    :param true_anchor:  top left x and y (x,y)
    :param true_hw:  anchor's width and height (h,w) use for calculate iou
    :return: general iou distribution without any hyperparameter for attention loss
    """
    def cal_single_iou(box1, box2):
        smooth = 1e-7
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max((yi2 - yi1), 0.) * max((xi2 - xi1), 0.)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        iou = (inter_area + smooth) / (union_area + smooth)
        return iou
    FEAT_WIDTH = grid_shapes[1]
    FEAT_HEIGHT = grid_shapes[0]


    t_h, t_w=true_hw
    t_x,t_y=true_xy

    gt_box=[t_x, t_y,t_x+t_w,t_y+t_h]

    iou_map=np.zeros([FEAT_HEIGHT, FEAT_WIDTH])
    for i in range(FEAT_WIDTH):
        for j in range(FEAT_HEIGHT):
            iou_map[j,i]=cal_single_iou(gt_box,[max(i-t_w/2,0.),max(j-t_h/2,0.),min(i+t_w/2,FEAT_WIDTH),min(j+t_h/2,FEAT_HEIGHT)])

    return iou_map


def pad_object_maps(obj_maps, max_len):
    ori_len, h, w = obj_maps.shape
    padded_maps = torch.zeros((max_len, h, w)).to(obj_maps.device)
    padded_maps[:ori_len] = obj_maps
    return padded_maps

def visual_sample(img_path, bboxs, obj_maps, img_h, img_w, words):
    """ Visualize sample 
    """
    os.makedirs("results/dataset", exist_ok=True)
    img = cv2.imread(img_path)
    fig, ax = plt.subplots(len(bboxs))
    plt.figure(figsize=(6, len(bboxs) * 6))
    for i in range(len(bboxs)):
        heatmap = obj_maps[i]
        heatmap = cv2.resize(heatmap, (img_w, img_h), cv2.INTER_CUBIC)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        hm_img = heatmap*0.3 + img * 0.7
        hm_img = np.uint8(hm_img)
        xc, yc, w, h = bboxs[i]
        x1 = int((xc - w/2) * img_w)
        x2 = int((xc + w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        y2 = int((yc + h/2) * img_h)
        pts = [np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.int32).reshape(-1, 1, 2)]
        # hm_img = np.ascontiguousarray(hm_img[:, :, [2, 1, 0]], dtype=np.uint8)
        hm_img = cv2.cvtColor(hm_img, cv2.COLOR_BGR2RGB)
        hm_img = cv2.polylines(hm_img, pts, True, (255, 0, 0), thickness=2)
        ax[i].set_title(words[i])
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(hm_img)
    fig.savefig(f"results/dataset/{words[1]}.jpg", bbox_inches='tight',dpi=500)
    plt.close()
    print(words)
