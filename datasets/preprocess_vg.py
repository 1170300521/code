import json
import tqdm
import spacy
import copy

nlp = spacy.load('en_core_web_lg')

region_path = "Dataset/VisualGenome/region_graphs.json"
attr_path = "Dataset/VisualGenome/attributes.json"
trn_obj_path = "Dataset/VisualGenome/align_modal/train_region_graphs.json"
val_obj_path = "Dataset/VisualGenome/align_modal/val_region_graphs.json"

trn_sgg_path = "Dataset/VisualGenome/sgg/train_sgg.json"
val_sgg_path = "Dataset/VisualGenome/sgg/val_sgg.json"
relation_path  = "Dataset/VisualGenome/sgg/rel_categories.json"
attr_class_path = "Dataset/VisualGenome/sgg/attr_categories.json"

def get_bboxs(ids, annot):
    if len(ids) == 0:
        bboxs = None
    else:
        obj_annot = annot['objects']
        x1 = obj_annot[ids[0]]['x']
        y1 = obj_annot[ids[0]]['y']
        x2 = x1 + obj_annot[ids[0]]['w']
        y2 = y1 + obj_annot[ids[0]]['h']
        for i in range(1, len(ids)):
            x1 = min(x1, obj_annot[ids[i]]['x'])
            y1 = min(y1, obj_annot[ids[i]]['y'])
            x2 = max(x2, obj_annot[ids[i]]['x'] + obj_annot[ids[i]]['w'])
            y2 = max(y2, obj_annot[ids[i]]['y'] + obj_annot[ids[i]]['h'])
        # x1 = abs(x1 / img_w)
        # x2 = abs(x2 / img_w)
        # y1 = abs(y1 / img_h)
        # y2 = abs(y2 / img_h)
        # bboxs.append([(x1+x2)/2, (y1+y2)/2, abs(x2-x1), abs(y2-y1)])
        bboxs = {
            'x': x1,
            'y': y1,
            'w': abs(x2 - x1),
            'h': abs(y2 - y1)
        }
    return bboxs

def get_object_idx(obj_id,words2obj_ids):
    for w in words2obj_ids:
        if obj_id in words2obj_ids[w]:
            return w
    return -1

# Read data from files
with open(region_path, 'r') as f:
    region_data = json.load(f)
    rel_classes = set()
    for i in region_data:
        for j in i['regions']:
            for k in j['relationships']:
                rel_classes.add(k['predicate'].lower().strip())
    rel2ids = {k: i for i, k in enumerate(rel_classes)}
    region_data = {i['image_id']: i for i in region_data}
    with open(relation_path, 'w') as f1:
        json.dump(rel2ids, f1)
    print('Read region_graphs.json')

with open(attr_path, 'r') as f:
    attr_data = json.load(f)
    attr_classes = set()
    for i in attr_data:
        for j in i['attributes']:
            if 'attributes' not in j.keys():
                continue
            for k in j['attributes']:
                attr_classes.add(k.lower().strip())
    attr2ids = {k: i for i, k in enumerate(attr_classes)}
    attr_data = {i['image_id']: i for i in attr_data}
    with open(attr_class_path, 'w') as f2:
        json.dump(attr2ids, f2)
with open(trn_obj_path, 'r') as f:
    trn_obj_data = json.load(f)

with open(val_obj_path, 'r') as f:
    val_obj_data = json.load(f)

trn_sgg_data = []
val_sgg_data = []
for img in tqdm.tqdm(val_obj_data):
# for img in tqdm.tqdm(val_obj_data):
    img_id = img['image_id']
    new_img = {'image_id': img_id}
    regions = []
    graphs = region_data[img_id]['regions']
    graphs = {i['region_id']: i['relationships'] for i in graphs}
    for region in img['regions']:
        new_region = {k: region[k] for k in region if k not in ['objects']}
        phrase = 'ANS ' + region['phrase']
        qtmp = nlp(str(phrase))
        qtmp_words = [x.text.lower() for x in qtmp]
        qtmp_lemmas = [x.lemma_ for x in qtmp]
        words2objects = [[] for x in range(len(qtmp_words))]
        # words2idx = {}
        obj_num = 0
        # words2objects[0] = [0]  # for ref_vg
        # print(words2objects)
        for j, o in enumerate(region['objects']):
            # For composition words, we use the last word
            obj_name = o['name'].split()[-1].lower()
            if obj_name in qtmp_words:
                word_idx = qtmp_words.index(obj_name)
                words2objects[word_idx].append(j)
                # words2idx[obj_name] = word_idx
                obj_num += 1
            elif obj_name in qtmp_lemmas:
                word_idx = qtmp_lemmas.index(obj_name)
                words2objects[word_idx].append(j)
                # words2idx[obj_name] = word_idx
                obj_num += 1
        if obj_num == 0:
            continue
        # Get objects
        objects = []
        for obj_name in range(len(words2objects)):
            obj_ids = words2objects[obj_name]
            bbox = get_bboxs(obj_ids, region)
            if bbox is None:
                continue
            obj = copy.deepcopy(bbox)
            obj['idx'] = obj_name
            obj['name'] = qtmp_words[obj_name]
            # obj['idx'] = words2idx[obj_name]
            objects.append(obj)
        new_region['objects'] = objects
        words2obj_ids = {}
        for obj_name in range(len(words2objects)):
            ids = words2objects[obj_name]
            obj_ids = set()
            for i in ids:
                obj_ids.add(region['objects'][i]['object_id'])
            words2obj_ids[obj_name] = obj_ids
        # Get attributes
        attributes = []
        for obj_name in range(len(words2objects)):
            # sent_idx = words2idx[obj_name]
            obj_ids = words2obj_ids[obj_name]
            if len(obj_ids) == 0:
                continue
            objs = attr_data[img_id]['attributes']
            attrs = set()
            attr_ids = set()
            for o in objs:
                if 'attributes' not in o:
                    continue
                obj_id = o['object_id']
                if obj_id in obj_ids:
                    tmp_attrs = {i.lower().strip() for i in o['attributes']}
                    tmp_attr_ids = {attr2ids[i] for i in tmp_attrs}
                    attrs = attrs.union(tmp_attrs)
                    attr_ids = attr_ids.union(tmp_attr_ids)
            if len(attrs) > 0:
                attributes.append({
                    'sent_idx': obj_name,
                    'attrs': list(attrs),
                    'attr_ids': list(attr_ids)
                })
        new_region['attributes'] = attributes
        # Get relationships
        relationships = {}
        rels = graphs[region['region_id']]
        for r in rels:
            r_sub_id = r['subject_id']
            r_obj_id = r['object_id']
            r_class = r['predicate'].lower().strip()
            r_class_id = rel2ids[r_class]
            sub_idx = get_object_idx(r_sub_id, words2obj_ids)
            obj_idx = get_object_idx(r_obj_id, words2obj_ids)
            if sub_idx != -1 and obj_idx != -1:
                relationships[(obj_idx, sub_idx)] = {
                    'obj_idx': obj_idx,
                    'sub_idx': sub_idx,
                    'rel_idx': r_class_id,
                    'predicate': r_class,
                }
        relationships = [relationships[k] for k in relationships.keys()]
        new_region['relationships'] = relationships
        regions.append(new_region)
    new_img['regions'] = regions
    val_sgg_data.append(new_img)

with open(val_sgg_path, 'w') as f:
    json.dump(val_sgg_data, f)
for img in tqdm.tqdm(trn_obj_data):
# for img in tqdm.tqdm(val_obj_data):
    img_id = img['image_id']
    new_img = {'image_id': img_id}
    regions = []
    graphs = region_data[img_id]['regions']
    graphs = {i['region_id']: i['relationships'] for i in graphs}
    for region in img['regions']:
        new_region = {k: region[k] for k in region if k not in ['objects']}
        phrase = 'ANS ' + region['phrase']
        qtmp = nlp(str(phrase))
        qtmp_words = [x.text.lower() for x in qtmp]
        qtmp_lemmas = [x.lemma_ for x in qtmp]
        words2objects = [[] for x in range(len(qtmp_words))]
        # words2idx = {}
        obj_num = 0
        # words2objects[0] = [0]  # for ref_vg
        # print(words2objects)
        for j, o in enumerate(region['objects']):
            # For composition words, we use the last word
            obj_name = o['name'].split()[-1].lower()
            if obj_name in qtmp_words:
                word_idx = qtmp_words.index(obj_name)
                words2objects[word_idx].append(j)
                # words2idx[obj_name] = word_idx
                obj_num += 1
            elif obj_name in qtmp_lemmas:
                word_idx = qtmp_lemmas.index(obj_name)
                words2objects[word_idx].append(j)
                # words2idx[obj_name] = word_idx
                obj_num += 1
        if obj_num == 0:
            continue
        # Get objects
        objects = []
        for obj_name in range(len(words2objects)):
            obj_ids = words2objects[obj_name]
            bbox = get_bboxs(obj_ids, region)
            if bbox is None:
                continue
            obj = copy.deepcopy(bbox)
            obj['idx'] = obj_name
            obj['name'] = qtmp_words[obj_name]
            # obj['idx'] = words2idx[obj_name]
            objects.append(obj)
        new_region['objects'] = objects
        words2obj_ids = {}
        for obj_name in range(len(words2objects)):
            ids = words2objects[obj_name]
            obj_ids = set()
            for i in ids:
                obj_ids.add(region['objects'][i]['object_id'])
            words2obj_ids[obj_name] = obj_ids
        # Get attributes
        attributes = []
        for obj_name in range(len(words2objects)):
            # sent_idx = words2idx[obj_name]
            obj_ids = words2obj_ids[obj_name]
            if len(obj_ids) == 0:
                continue
            objs = attr_data[img_id]['attributes']
            attrs = set()
            attr_ids = set()
            for o in objs:
                if 'attributes' not in o:
                    continue
                obj_id = o['object_id']
                if obj_id in obj_ids:
                    tmp_attrs = {i.lower().strip() for i in o['attributes']}
                    tmp_attr_ids = {attr2ids[i] for i in tmp_attrs}
                    attrs = attrs.union(tmp_attrs)
                    attr_ids = attr_ids.union(tmp_attr_ids)
            if len(attrs) > 0:
                attributes.append({
                    'sent_idx': obj_name,
                    'attrs': list(attrs),
                    'attr_ids': list(attr_ids)
                })
        new_region['attributes'] = attributes
        # Get relationships
        relationships = {}
        rels = graphs[region['region_id']]
        for r in rels:
            r_sub_id = r['subject_id']
            r_obj_id = r['object_id']
            r_class = r['predicate'].lower().strip()
            r_class_id = rel2ids[r_class]
            sub_idx = get_object_idx(r_sub_id, words2obj_ids)
            obj_idx = get_object_idx(r_obj_id, words2obj_ids)
            if sub_idx != -1 and obj_idx != -1:
                relationships[(obj_idx, sub_idx)] = {
                    'obj_idx': obj_idx,
                    'sub_idx': sub_idx,
                    'rel_idx': r_class_id,
                    'predicate': r_class,
                }
        relationships = [relationships[k] for k in relationships.keys()]
        new_region['relationships'] = relationships
        regions.append(new_region)
    new_img['regions'] = regions
    trn_sgg_data.append(new_img)

with open(trn_sgg_path, 'w') as f:
    json.dump(trn_sgg_data, f)