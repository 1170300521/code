"""
RDETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import activation
from transformers import RobertaModel, RobertaTokenizerFast

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, IoU_values)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer, FeatureResizer
from .position_encoding import WordPositionEmbeddingSine
from .vilbert import BertLMPredictionHead


CORRECT_IOUS = []
WRONG_IOUS = []


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, 
                 query_pos='sine', matcher='hungarian', is_pretrain=False, bert_type=None,
                 use_mlm=False, no_img=False, no_obj_att=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            query_pos: position encoding for language query
            matcher: method of matching between gt and prediction bboxes
            is_pretrain: whether to pretrain or not
            bert_type: pretrained model of bert
        """
        super().__init__()
        self.is_pretrain = is_pretrain
        self.num_queries = num_queries
        self.transformer = transformer
        self.bert_type = bert_type
        self.use_mlm = use_mlm
        self.no_img = no_img
        self.no_obj_att = no_obj_att
        hidden_dim = transformer.d_model
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Map language to hidden_dim
        query_proj_dim = 300 if self.bert_type is None else 768
        if self.bert_type is None:
            self.query_proj = nn.Linear(query_proj_dim, hidden_dim)
        else:
            self.query_proj = FeatureResizer(query_proj_dim, hidden_dim, 0.1)
        # Map image to hidden_dim
        if not self.no_img:
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        # Bert language tokenizer and model
        if self.bert_type is not None:
            self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_type)
            self.bert_model = RobertaModel.from_pretrained(self.bert_type)
            # Freeze bert parameters during training
            for p in self.bert_model.parameters():
                p.requires_grad_(False)
        if self.use_mlm:
            self.mlm_pred = BertLMPredictionHead(hidden_dim, class_num=30522, activation='relu')
        
        self.aux_loss = aux_loss and not is_pretrain
        if self.is_pretrain:
            if not self.no_img:
                self.match_pred = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, 2)
                )
            self.text_pred = BertLMPredictionHead(hidden_dim, class_num=30522, activation='relu')
        else:
            if not self.no_obj_att:
                self.visual_sim = nn.Linear(hidden_dim, hidden_dim)
                self.text_sim = nn.Linear(hidden_dim, hidden_dim)
            self.class_emb = nn.Linear(hidden_dim, num_classes + 1) if matcher != 'first' else None
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.attr_emb = nn.Linear(hidden_dim, 66000)
            self.rel_emb = MLP(hidden_dim * 2, hidden_dim * 2, 40000, 3)
        if query_pos == 'learned':
            self.query_pos = nn.Embedding(num_queries, hidden_dim)
        elif query_pos == 'sine':
            self.query_pos = WordPositionEmbeddingSine(num_queries, hidden_dim)
        else:
            self.query_pos = None
    
    def forward(self, samples: NestedTensor, word_emb, targets, visualize=False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
               word_emb: NestedTensor or str

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        src = None
        pos = None
        mask = None
        if not self.no_img:
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            device = src.device
            _, _, h, w = src.shape
            assert mask is not None
            pos = pos[-1]
            src = self.input_proj(src)
        if self.bert_type is None:
            query, lang_mask = word_emb.decompose()
        else:
            tokens = self.tokenizer.batch_encode_plus(word_emb, padding='longest', return_tensors='pt').to(device)
            query = self.bert_model(**tokens)
            query = query.last_hidden_state
            # print(query.shape)
            lang_mask = tokens.attention_mask.ne(1).bool()
        query = self.query_proj(query)
        # lang_mask = F.interpolate(lang_mask[None].float(), size=query.shape[-1:]).to(torch.bool)[0]
        query_pos = self.query_pos.weight if self.query_pos is not None else None
        # Roberta Bert forward func has added pos to word embedding
#        if self.bert_type is not None:
#            query_pos = torch.zeros_like(query_pos)
        b, l, _ = query.shape
        query_pos = query_pos[:l]
        visual_dict, _ = self.transformer(query, src, mask, query_pos, pos, lang_mask)
        if not self.is_pretrain:
            hs = visual_dict['hs']
            outputs_coord = self.bbox_embed(hs).sigmoid()
            outputs_class = self.class_emb(hs) if self.class_emb is not None else outputs_coord 
            d_num = len(outputs_class)
            if len(targets['batch_attr']) == 0:
                outputs_attr = [None] * d_num
            else:
                outputs_attr = self.attr_emb(hs[:, targets['batch_attr'], targets['attr_ids']])
            if len(targets['batch_rel']) == 0:
                outputs_rel = [None] * d_num
            else:
                sub_hs = hs[:, targets['batch_rel'], targets['sub_ids']]
                obj_hs = hs[:, targets['batch_rel'], targets['obj_ids']]
                outputs_rel = self.rel_emb(torch.cat([obj_hs, sub_hs], dim=-1))
            out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],\
                'pred_attrs': outputs_attr[-1], 'pred_rels': outputs_rel[-1]}
            if not self.no_obj_att:
                cross_lang = self.text_sim(visual_dict['cross_lang']).unsqueeze(2)  # b * L * 1 * c
                cross_img = self.visual_sim(visual_dict['cross_img']).unsqueeze(1)  # b * 1 * (h*w) * c
                pred_obj_att = (cross_lang * cross_img).sum(dim=-1).view(b, l, h, w)
                out['pred_obj_att'] = pred_obj_att
           
            if self.use_mlm:
                cross_lang = visual_dict['cross_lang']
                out['text_pred'] = self.mlm_pred(cross_lang)
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,\
                    outputs_attr, outputs_rel)
            if visualize:
                out['self_att'] = visual_dict['self_att'].cpu().detach().transpose(0, 1).tolist()  # NxBxLxS -> BxNxLxS
                out['cross_att'] = visual_dict['cross_att'].cpu().detach().transpose(0, 1).tolist()
                out['size'] = [h, w]
        else:
            cross_lang = visual_dict['cross_lang']
            if not self.no_img:
                match_pred = self.match_pred(cross_lang[:, 0])
                out = {
                    "match_pred": match_pred,
                    "cross_att": visual_dict['cross_att']  # H * B * L * h * w
                }
            else:
                out = {}
            text_pred = self.text_pred(cross_lang)
            out['text_pred'] = text_pred
            
        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_attr, outputs_rel):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_attrs': c, 'pred_rels': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], \
                    outputs_attr[:-1], outputs_rel[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, is_pretrain=False):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.acc_iou_threshold = 0.5
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.is_pretrain = is_pretrain
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t[J][:, 0] for t, (_, J) in zip(targets['labels'].tensors, indices)])
        tgt_ids = self._get_tgt_permutation_idx(indices)
        target_classes_o = targets['labels'].tensors[tgt_ids][:, 0]
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v) for v in targets['labels']], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_accuracy(self, outputs, targets, indices, num_boxes):
        """ Compute the accuracy """
        # pred_logits = outputs['pred_logits'][:, :, 0]  # '0' rep objects
        # _, ids = pred_logits.max(1)
        ids = self._get_src_permutation_idx(indices)
        tgt_ids = self._get_tgt_permutation_idx(indices)
        # target_bboxs = box_ops.box_cxcywh_to_xyxy(targets['cthw'].tensors[dtgt_ids])
        results = PostProcess()(outputs['pred_boxes'], targets['orig_size'])
        gt = PostProcess()(targets['cthw'].tensors, targets['orig_size'])
        # ids = ids.to(results.device)
        # pred_boxes = torch.gather(results, 1, ids.view(-1, 1, 1).expand(-1, 1, 4))
        # pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = results[ids]
        target_bboxs = gt[tgt_ids]
        ious = torch.diag(IoU_values(pred_boxes, target_bboxs))
        CORRECT_IOUS.extend(ious[ious >= self.acc_iou_threshold].tolist())
        WRONG_IOUS.extend(ious[ious < self.acc_iou_threshold].tolist())
        # import pdb
        # pdb.set_trace()
        if torch.isnan(ious).sum() > 0 or torch.isnan((ious >= self.acc_iou_threshold).float().mean()):
            import pdb
            pdb.set_trace()
        return {
            "accuracy": (ious >= self.acc_iou_threshold).float().mean(),
        }

    @torch.no_grad()
    def loss_mlm_acc(self, outputs, targets):
        text_pred = outputs['text_pred']
        text_labels = targets['text_labels']
        B, T, _ = text_pred.shape
        text_pred = text_pred.view(B * T, -1)
        text_labels = text_labels.view(B * T)
        _, ids = text_pred.max(1)
        num_pred = (text_labels != -1).float().sum() + 1e-6
        return {"mlm_acc": (ids == text_labels).float().sum() / num_pred}

    @torch.no_grad()
    def loss_match_acc(self, outputs, targets):
        match_pred = outputs['match_pred']
        is_match = targets['is_match']
        _, ids = match_pred.max(1)
        return {"match_acc": (ids == is_match).float().mean()}        

    def loss_mlm(self, outputs, targets):
        text_pred = outputs['text_pred']
        text_labels = targets['text_labels']
        B, T, _ = text_pred.shape
        text_pred = text_pred.view(B * T, -1)
        text_labels = text_labels.view(B * T)
        loss = F.cross_entropy(text_pred + 1e-8, text_labels, ignore_index=-1)
        # torch.clip_(loss, min=1e-6, max=20)
        return {"loss_mlm": loss}

    def loss_match(self, outputs, targets):
        match_pred = outputs['match_pred']
        is_match = targets['is_match']
        # 0 reps match
        return {"loss_match": F.cross_entropy(match_pred, is_match)}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        tgt_ids = self._get_tgt_permutation_idx(indices)
        #target_boxes = torch.cat([t[i] for t, (_, i) in zip(targets['cthw'].tensors, indices)], dim=0)
        target_boxes = targets['cthw'].tensors[tgt_ids]
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_obj_att(self, outputs, targets, indices, num_bboxs):
        ids = self._get_src_permutation_idx(indices)
        pred_att = outputs['pred_obj_att']
        gt_att = targets['obj_maps']
        # seq_mask = (targets['labels'].tensors == 0).float().unsqueeze(-1)
        # bs, seq_len = targets['labels'].mask.shape
        # seq_mask = (~targets['labels'].mask & (targets['labels'].tensors.view(bs, seq_len) == 0)).float()
        # seq_mask = seq_mask.view(bs, seq_len, 1, 1)
        # att_mask = ((~gt_att.mask.unsqueeze(1)).float() * seq_mask).bool()
        pred_att = pred_att[ids]
        gt_obj_att = gt_att.tensors[ids]
        # att_mask = F.interpolate(att_mask, size=pred_att.shape[-2:]).to(torch.bool)
        # gt_obj_att = F.interpolate(gt_obj_att, size=pred_att.shape[-2:])
        losses = {
            "loss_obj_att": F.binary_cross_entropy_with_logits(pred_att, \
                gt_obj_att),
                # gt_obj_att, weight=att_mask),
        }
        return losses

    def loss_attr(self, outputs, targets, indices, num_bboxs):
        pred_attr = outputs['pred_attrs']
        gt_attr = targets['attr_labels']
        if pred_attr is None:
            return {
                "loss_attr": torch.tensor(0.).to(outputs['pred_logits'].device)
            }
        return {
            "loss_attr": F.binary_cross_entropy_with_logits(pred_attr, gt_attr)
        }
    
    def loss_rel(self, outputs, targets, indices, num_bboxs):
        pred_rel = outputs['pred_rels']
        gt_rel = targets['rel_labels']
        if pred_rel is None:
            return {
                "loss_rel": torch.tensor(0.).to(outputs['pred_logits'].device)
            }
        return {
            "loss_rel": F.cross_entropy(pred_rel, gt_rel,)
        }
    @torch.no_grad()
    def loss_rel_acc(self, outputs, targets, indices, num_bboxs):
        if outputs['pred_rels'] is None:
            return {
                "loss_rel_acc": torch.tensor(0.).to(outputs['pred_logits'].device)
            }
        _, pred_rel = outputs['pred_rels'].max(1)
        return {
            "loss_rel_acc": (pred_rel == targets['rel_labels']).float().mean()
        }
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices=None, num_boxes=None, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'accuracy': self.loss_accuracy,
            'mlm': self.loss_mlm,
            'match': self.loss_match,
            'mlm_acc': self.loss_mlm_acc,
            'match_acc': self.loss_match_acc,
            'obj_att': self.loss_obj_att,
            'attr': self.loss_attr,
            'rel': self.loss_rel,
            'rel_acc': self.loss_rel_acc,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        # Whether to pretrain model
        if indices is not None:
            return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
        else:
            return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ['aux_outputs']}
        losses = {}
        indices = None
        num_boxes = None
        if not self.is_pretrain:
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            # num_boxes = sum(len(t) for t in targets['labels'].tensors)
            # num_boxes = (targets['labels'].tensors == 0).float().sum()
            # num_boxes = targets['boxes'].size(0)
            num_boxes = sum(len(i) for (i, j) in indices)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        # Compute all the requested losses
        for loss in self.losses:
            if loss in ['mlm', 'match', 'mlm_acc', 'match_acc']:
                losses.update(self.get_loss(loss, outputs_without_aux, targets))
            else:
                losses.update(self.get_loss(loss, outputs_without_aux, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['masks', 'mlm', 'mlm_acc', 'obj_att']:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, out_bbox, target_sizes):
        """ Perform the computation
        Parameters:
            out_box: normalized bboxs, cthw
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # out_bbox.shape

        # assert len(out_logits) == len(target_sizes)
        # assert target_sizes.shape[1] == 2

        # prob = F.softmax(out_logits, -1)
        # scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return boxes


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    num_classes = 1
    device = torch.device(args.device)
    if args.no_img:
        backbone = None
    else:
        backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        query_pos=args.query_pos,
        matcher=args.matcher,
        is_pretrain=(args.ds_name == 'pretrain'),
        bert_type=args.bert_type,
        use_mlm=args.use_mlm,
        no_img=args.no_img,
        no_obj_att=args.no_obj_att
    )

    is_pretrain = args.ds_name == 'pretrain'
    if not is_pretrain:
        matcher = build_matcher(args)
        weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
        weight_dict['loss_giou'] = args.giou_loss_coef
        weight_dict['loss_attr'] = args.attr_loss_coef
        weight_dict['loss_rel'] = args.rel_loss_coef
        # TODO this is a hack
        if args.aux_loss:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        
        if args.matcher == 'first':
            # we only need the first output, so ignore labels and cardinality
            losses = ['boxes', 'accuracy']
        else:
            losses = ['labels', 'boxes', 'accuracy', 'attr', 'rel', 'rel_acc']
        if not args.no_obj_att:
            weight_dict['loss_obj_att'] = args.obj_att_coef
            losses.append('obj_att')
        if args.use_mlm:
            losses.append('mlm')
            losses.append('mlm_acc')
            weight_dict['loss_mlm'] = args.mlm_loss_coef
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                eos_coef=args.eos_coef, losses=losses)
        postprocessors = {'bbox': PostProcess()}
    else:
        weight_dict = {
            'loss_mlm': args.mlm_loss_coef,
        }
        losses = ['mlm', 'mlm_acc']
        if not args.no_img:
            weight_dict['loss_match'] = args.match_loss_coef
            losses.extend(['match', 'match_acc'])

        criterion = SetCriterion(num_classes=0, matcher=None, weight_dict=weight_dict,
                                eos_coef=-1, losses=losses, is_pretrain=is_pretrain)
        postprocessors = None
    criterion.to(device)

    return model, criterion, postprocessors
