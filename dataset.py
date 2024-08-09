import copy
import json
import logging
import math
import os
import io
import gc

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import numpy as np
import torch
import transformers
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
import traceback
from copy import deepcopy
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from .conversation import get_conv_template, Conversation
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from .utils import get_frame_indices, read_frames_gif, read_frames_decord, read_frames_folder, decode_video
from .constants import (CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
                        IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN,
                        SIGLIP_MEAN, SIGLIP_STD)

from transformers.trainer_pt_utils import LabelSmoother
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
IGNORE_INDEX = -100
logger = logging.getLogger(__name__)

'''
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained('/data/model/InternVL2-2B',trust_remote_code=True)
from dataset import SupervisedDataset
tokenizer = AutoTokenizer.from_pretrained('/data/model/InternVL2-2B', trust_remote_code=True)
data = SupervisedDataset(template_name=model.template, meta={'annotation':'/data/code/videotrack/anno_file/sav_train.jsonl'},ds_name='sav',tokenizer=tokenizer,num_image_token=model.num_image_token,)
for x in data:
    pass
s = data.__getitem__(0)
'''
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        ds_name,
        tokenizer,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=True,
        use_thumbnail=True,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=4,  # for video data
        max_num_frame=12,  # for video data
        num_frames = 8,
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        random_seed=0,
        annot_sample_rate=4,
    ):
        super(SupervisedDataset, self).__init__()
        self.count_1 = 0
        self.count_2 = 0
        self.count_3 = 0 
        self.ds_name = ds_name
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method 
        self.num_frames = num_frames
        
        self.tokenizer = tokenizer
        self.use_thumbnail  = use_thumbnail 
        self.dynamic_image_size = dynamic_image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.transform = build_transform(
            self.image_size,
            is_train,
        )
        self.vis_transform = build_vis_transform(
            self.image_size,
        )
        # self.num_segments = num_segments 
        self.template_name = template_name
        self.group_by_length = group_by_length
        self.num_image_token = num_image_token

        self.annot_sample_rate = 4

        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'
        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        with open('/mnt/data/user/tc_agi/multi_modal/cqy/training_data/videotrack/template/sot_template.txt', 'r') as f:
            self.sot_template = f.readlines()
        
        gc.collect()
        
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    
    def __len__(self):
        return len(self.raw_data)

    def multi_modal_get_item(self, data_item):
        pass
        # pixel_values, num_patches_list, query = preprocess_visual(data_item['path'], self.transform, 
        #                                                           self.num_segments, data_item['is_image'], 
        #                                                           self.dynamic_image_size,
        #                                                         )
        # if '<image>' not in data_item['conversations'][0]['value']:
        #     data_item['conversations'][0]['value'] = query + data_item['conversations'][0]['value']
        # num_patches = num_patches_list[0]
        # total_num_patches = sum(num_patches_list)
        # if not self.dynamic_image_size:
        #     assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'
        # ret = preprocess_internlm(self.template_name, [deepcopy(data_item['conversations'])],
        #                           self.tokenizer, self.num_image_token * num_patches,
        #                           group_by_length=self.group_by_length, ds_name=data_item['ds_name'],
        #                           num_image = len(num_patches_list)
        #                          )
        # ret = dict(
        #     input_ids=ret['input_ids'][0],
        #     labels=ret['labels'][0],
        #     attention_mask=ret['attention_mask'][0],
        #     pixel_values=pixel_values,
        #     image_flags=torch.tensor([total_num_patches], dtype=torch.long)
        # )
        return ret
        
    def multi_modal_multi_image_get_item(self, data_item):
        pass
        # images, num_tiles = [], []
        # num_image = len(data_item['image'])
        # for image_path in data_item['image']:
        #     # Merge the image path
        #     image_path = self.get_image_path(image_path)
        #     # Load the image using tcs_loader if available, otherwise use PIL
        #     image = self.load_image(image_path)
        #     if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
        #         image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
        #                                    max_num=self.max_dynamic_patch // num_image,
        #                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        #         images += image
        #         num_tiles.append(len(image))
        #     else:  # Otherwise, use the original image as a single patch
        #         images.append(image)
        #         num_tiles.append(1)
        # pixel_values = [transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)
        # num_patches = pixel_values.size(0)

        # # Select the appropriate preprocessing function based on the template name
        # preprocess_function = self.get_preprocess_function()

        # # Preprocess the conversations and generate the return dictionary
        # num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        # ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
        #                           self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
        #                           ds_name=self.ds_name, num_image=num_image)

        # # Create the final return dictionary
        # ret = dict(
        #     input_ids=ret['input_ids'][0],
        #     labels=ret['labels'][0],
        #     attention_mask=ret['attention_mask'][0],
        #     pixel_values=pixel_values,
        #     image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        # )
        return ret
        
    def video_get_item(self, data_item):
        # def sample_starting_frame(total_frames, num_segments):
        #     # 将总帧数分成 num_segments 份
        #     intervals = np.linspace(0, total_frames, num_segments + 1).astype(int)
        #     first_segment_range = (intervals[0], intervals[1] - 1)
            
        #     # 从第一份中随机采样起始帧的位置
        #     start_frame_index = random.randint(first_segment_range[0], first_segment_range[1])
            
        #     return start_frame_index
            
        # masklet_index = [x for x in range(data_item['masklet_frame_count'])]
        # start_index = sample_starting_frame(data_item['masklet_frame_count'], self.num_frames)

        all_bbox_list = [sum(data_item['masklet'][index]['bbox']) > 0 for index in range(len(data_item['masklet']))]
        box_pos_list = [index for index, item in enumerate(all_bbox_list) if item]
        bbox_pos_list = [data_item['masklet'][index]['bbox'] for index in box_pos_list]
        if len(box_pos_list) > 0:
            self.count_3+=1
        image_list, frame_indices = self.preprocess_visual(
            # data_item['video_path'].replace('/data/data/video_track/sav/','/home/jeeves/'),
            data_item['video_path'],
            image_type='video',
            start_index=data_item['masklet_first_appeared_frame'],
            box_pos_list=box_pos_list
        )
        # vis_img = []
        # # os.system(f"cp {data_item['video_path'].replace('/data/data/video_track/sav/','/home/jeeves/')} ./")
        # for bbox, img in zip([bbox_pos_list[index] for index in frame_indices], image_list):
        #     vis_img.append(draw_bounding_boxes(img.convert('RGB'), [bbox], 'tar'))
        # fps = 10 
        # name = data_item['video_path'].split('/')[-1].split('.')[0]
        # vis_img[0].save(
        #     f"{name}.gif", 
        #     save_all = True,
        #     append_images=vis_img[1:],
        #     duration=int(1000 / fps),
        #     loop=0  # 设置为 0 表示无限循环
        # )
        # image_list[0].save(
        #     f"{name}_raw.gif", 
        #     save_all = True,
        #     append_images=image_list[1:],
        #     duration=int(1000 / fps),
        #     loop=0  # 设置为 0 表示无限循环
        # )
        # Transform each frame image and stack them into a tensor
        original_width, original_height = image_list[0].size
        # 缩放比例
        scale_x = self.image_size / original_width
        scale_y = self.image_size / original_height

        bbox_list = self.resize_bbox([bbox_pos_list[index] for index in frame_indices], scale_x, scale_y)
        prompt, answer = self.make_sot_seq(bbox_list, )
        
        conversations = []
        
        conversations.append(
            {
                'from': 'human',
                'value': prompt,
            }
        )
        conversations.append(
            {
                'from': 'gpt',
                'value': answer,
            }
        )
        # Ensure the first conversation contains a video placeholder
        if '<video>' not in conversations[0]['value']:
            conversations[0]['value'] = '<video>\n' + conversations[0]['value']
        special_tokens = '\n'.join(['Frame{}: <image>'.format(i + 1) for i in range(len(image_list))])
        conversations[0]['value'] = conversations[0]['value'].replace('<video>\n', special_tokens)
        images, num_tiles = [], []
        num_image = len(image_list)
        for image in image_list:
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // 3,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        # import pdb
        # pdb.set_trace()
        # print(conversations)
        ret = preprocess_internlm(self.template_name, [deepcopy(conversations)],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  ds_name=self.ds_name, num_image=num_image)
        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def resize_bbox(self, bbox_list, scale_x, scale_y):
        # 缩放后的bbox列表
        scaled_bboxes = []
        for bbox in bbox_list:
            scaled_bbox = [
                bbox[0] * scale_x ,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ]
            scaled_bboxes.append(scaled_bbox)
        
        return scaled_bboxes
        
    def make_sot_seq(self, bbox_list):
        prompt = random.choice(self.sot_template)
        bbox_norm = []
        bbox_quali = []
        index_list = []
        for  index, bbox in enumerate(bbox_list):
            scaled_bbox = [
                int(bbox[0] * 1000 / self.image_size) ,
                int(bbox[1] * 1000 / self.image_size) ,
                int(bbox[2] * 1000 / self.image_size) ,
                int(bbox[3] * 1000 / self.image_size) ,
            ]
            if sum(scaled_bbox)>0:
                bbox_quali.append(1)
                bbox_norm.append(scaled_bbox)
                index_list.append(index)
            else:
                bbox_quali.append(0)
                bbox_norm.append([0,0,0,0])
        # if sum(bbox_quali)<4:
        #     print(bbox_list)
        assert sum(bbox_quali) >= 4
        begin_index = random.choice(index_list)
        init_bbox = bbox_norm[begin_index]
        init_bbox_seq = f"<box>[[{init_bbox[0]}, {init_bbox[1]}, {init_bbox[2]}, {init_bbox[3]}]]</box>"
        prompt = prompt.replace('<index>', str(begin_index + 1)).replace('<box>', init_bbox_seq)
        
        answer = ','.join([f" Frame {index+1}: <box>[[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]]</box>" for index, bbox in enumerate(bbox_norm)])
        return prompt, answer
        
        
    def preprocess_visual(self, visual_path, image_type='image', start_index=None, box_pos_list=None):
        if image_type == 'image':
            pixel_values = load_image(visual_path, self.transform, max_num=self.max_num, is_dynamic_preprocess=False)
            # num_patches_list = [pixel_values.shape[0]]
            # prefix = "Frame1: <image>\n" 
            frame_indices = [0]
            
        elif image_type == 'multi_image':
            image_list = sorted(list(os.listdir(visual_path)))
            
            pixel_values = []
            num_patches_list = []
            for path in visual_path:
                pixel_values_part = load_image(path, self.transform, max_num=max_num, is_dynamic_preprocess=False)
                # num_patches_list.append(pixel_values_part.size(0))
                pixel_values.append(pixel_values_part)
                
            vlen = len(pixel_values)
            if vlen > self.num_frames:
                frame_indices = get_frame_indices(
                                    self.num_frames, vlen, sample=self.sampling_method, 
                                )
            # pixel_values = [pixel_values[i] for i in frame_indices]
            # prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        elif image_type == 'video':
            pixel_values, frame_indices = read_frames_decord(
                visual_path, self.num_frames, sample=self.sampling_method, 
                start_index=start_index, sample_rate=self.annot_sample_rate,
                box_pos_list=box_pos_list,
            )
            
            # pixel_values_list, num_patches_list = [], []
            # for img in frames:
            #     if self.dynamic_image_size:
            #         img = dynamic_preprocess(img, image_size=self.input_size, use_thumbnail=True, max_num=self.max_dynamic_patch)
            #     pixel_values = [self.transform(tile) for tile in img]
            #     pixel_values = torch.stack(pixel_values)
            #     num_patches_list.append(pixel_values.shape[0])
            #     pixel_values_list.append(pixel_values)
            # pixel_values = torch.cat(pixel_values_list)
            # prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            
        return pixel_values, frame_indices
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # try:
        #     ret = self.multi_modal_get_item(self.raw_data[i])
            
        # '''
        # {
        # is_image:'single_image','single2multi_image','multi_image','video'
        # 'visual_path':
        # 'conversations': [
        # {'user':question},
        # {"assistant":answer},
        # ]
        # }
        # '''
        # except:
        #     print(traceback.format_exc(),1)
        #     return self.__getitem__(random.randint(0, len(self)))
        i = i % len(self.raw_data)
        while True:
        # if 1:
            try:
            
            # if 1:
                self.count_1 +=1
                data_item = copy.deepcopy(json.loads(self.raw_data[i]))
                if data_item['data_type'] == 'multi_image':
                    ret = self.multi_modal_multi_image_get_item(data_item)
                elif data_item['data_type'] == 'image':
                    ret = self.multi_modal_get_item(data_item)
                elif data_item['data_type'] == 'video':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                self.count_2+=1
                print(e, self.ds_name, flush=True)
                # if not isinstance(e, UnidentifiedImageError):
                #     traceback.print_exc()
                # data_item = json.loads(self.raw_data[i])
                # if data_item['data_type'] == 'multi_image':
                #     images = os.listdir(data_item['visual_path'])
                #     print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                # elif data_item['data_type'] == 'image':
                #     data_path = data_item['visual_path']
                #     print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                # elif 'video' in data_item:
                #     data_path = data_item['video_path']
                #     print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        # print(self.count_2/self.count_1, self.count_3/self.count_1)
        # import pdb
        # pdb.set_trace()
        return ret

def concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if 'label' in first and first['label'] is not None:
        label = first['label'].item() if isinstance(first['label'], torch.Tensor) else first['label']
        dtype = torch.long if isinstance(label, int) else torch.float
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=dtype)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            dtype = torch.long if isinstance(first['label_ids'][0], int) else torch.float
            batch['labels'] = torch.tensor([f['label_ids'] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and \
                v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')


class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn, image_type='image', max_num_frames=-1, min_num_frames=4, sample='rand', clip=None):
        if image_type == 'image':
            img_value_str = self.client.get(fn)
            img = pil_loader(img_value_str)
            return img

        elif image_type == 'video':
            if fn.endswith('/'):
                frames = read_frames_folder(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            client=self.client, sample=sample)
            elif fn.endswith('.gif'):
                frames = read_frames_gif(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                         client=self.client, sample=sample)
            else:
                frames = read_frames_decord(fn, num_frames=max_num_frames, min_num_frames=min_num_frames,
                                            client=self.client, sample=sample, clip=clip)
            return frames


# def data_collator(examples, max_length=8192, tokenizer=None):
#     def trim_and_pad(seq, batch_first, padding_value):
#         return pad_sequence([s[:max_length] for s in seq], batch_first=True, padding_value=padding_value)
#     input_ids = trim_and_pad(
#         [example["input_ids"] for example in examples],
#         batch_first=True,
#         padding_value=tokenizer.pad_token_id,
#     )
    
#     targets = trim_and_pad(
#         [example["labels"] for example in examples],
#         batch_first=True,
#         padding_value=-100,
#     )
#     attention_mask = trim_and_pad(
#         [example["attention_mask"] for example in examples],
#         batch_first=True,
#         padding_value=tokenizer.unk_token_id,
#     )
#     pixel_values = [example["pixel_values"] for example in examples]
#     image_flags= [example["image_flags"] for example in examples]
#     return {
#         "input_ids": input_ids,
#         "labels": targets,
#         "attention_mask": attention_mask,
#         "image_flags": image_flags,
#         "pixel_values": pixel_values,
#     }



    

def simulate_jpeg_degradation(quality):
    def jpeg_degrade(img):
        with io.BytesIO() as output:
            img.convert('RGB').save(output, format='JPEG', quality=quality)
            output.seek(0)  # Move the reading cursor to the start of the stream
            img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
        return img_jpeg
    return jpeg_degrade


# Define the JPEG compression quality range, pre-create all JPEG compression functions
qualities = list(range(75, 101))
jpeg_degrade_functions = {quality: simulate_jpeg_degradation(quality) for quality in qualities}


def build_transform(input_size, is_train=False):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomChoice([T.Lambda(jpeg_degrade_functions[quality]) for quality in qualities]),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    else:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
    return transform

def build_vis_transform(input_size):
    return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])
    
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, transform, input_size=448, max_num=6, is_dynamic_preprocess=True):
    image = Image.open(image_file).convert('RGB')
    if is_dynamic_preprocess:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, transform, bound=None, input_size=448, max_num=1, num_segments=32, is_dynamic_preprocess=True):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    pixel_values_list, num_patches_list = [], []
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        if is_dynamic_preprocess:
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    conv = get_conv_template(template_name)
    roles = {'human': conv.roles[0], 'gpt': conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]['from']] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence['from']]
            assert role == conv.roles[j % 2], f'{i}'
            sentence['value'] = sentence['value'].strip()
            conv.append_message(role, sentence['value'])
        conversations.append(conv.get_prompt())

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                # print(i, len(num_image_token_list), num_image)
                image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[i]}{IMG_END_TOKEN}'
                conversation = conversation.replace('<image>', image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors='pt',
        # padding=False if group_by_length or use_packed_ds else 'max_length',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            print(repr(tokenizer.decode(z)))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f'WARNING: tokenization mismatch: {cur_len} vs. {total_len}. This dataset is {ds_name}.')
                sys.stdout.flush()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

from PIL import Image
import torchvision.transforms as T
import numpy as np
import imageio

def apply_transformation(image, transformation):
    return transformation(image)

def create_pseudo_video(image, bounding_box, frames=30):
    video_frames = []

    # Define transformation parameters
    max_translate = 20  # Maximum translation in pixels
    max_scale = 0.5  # Maximum scale change

    x_min, y_min, x_max, y_max = bounding_box
    width, height = x_max - x_min, y_max - y_min

    for i in range(frames):
        # Calculate translation for this frame
        translate_x = int(np.sin(i / frames * 2 * np.pi) * max_translate)
        translate_y = int(np.cos(i / frames * 2 * np.pi) * max_translate)

        # Calculate scale for this frame
        scale = 1 + max_scale * (i / frames)

        # Create transformation
        transformation = T.Compose([
            T.Resize((int(height * scale), int(width * scale))),
            T.RandomAffine(degrees=0, translate=(translate_x / width, translate_y / height))
        ])

        # Crop to the bounding box
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        
        # Apply transformation to the cropped image
        transformed_image = apply_transformation(cropped_image, transformation)
        
        # Crop or pad to original bounding box size
        if transformed_image.width > width or transformed_image.height > height:
            left = (transformed_image.width - width) // 2
            top = (transformed_image.height - height) // 2
            right = left + width
            bottom = top + height
            transformed_image = transformed_image.crop((left, top, right, bottom))
        else:
            padding = (width - transformed_image.width, height - transformed_image.height)
            transformed_image = T.Pad(padding)(transformed_image)

        # Add the transformed image to the video frames list
        video_frames.append(transformed_image)

    return video_frames


import os
import argparse
import torch
import json

from tqdm import tqdm
import os.path as osp
import torch
from torchvision.transforms import ToPILImage, PILToTensor
from torchvision.ops import box_convert
from torchvision.ops.boxes import box_area
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import requests
import re


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def draw_bounding_boxes(image, boxes, labels=None, colors=['red'], **kwargs):
    if isinstance(image, Image.Image):
        image = PILToTensor()(image) / 255.0
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    return _draw_bounding_boxes(image, boxes, labels=labels, colors=colors, **kwargs)


def _draw_bounding_boxes(image, boxes, labels=None, colors=None, **kwargs):
    assert isinstance(image, torch.Tensor) and isinstance(boxes, torch.Tensor)

    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype('uint8')
    img = Image.fromarray(image)

    draw = ImageDraw.Draw(img)

    for i in range(len(boxes)):
        box = boxes[i].tolist()
        color = colors[i]
        draw.rectangle(box, outline=color, width=3)

        if labels is not None and i < len(labels):
            label = labels[i]
            font = ImageFont.load_default()  # You can choose your own font here
            draw.text((box[0], box[1]), label, fill=color, font=font)

    # return ToPILImage()(img)
    return img



# # Example usage
# image_path = 'your_image_path.jpg'
# image = Image.open(image_path)
# bounding_box = [50, 50, 150, 150]  # Example bounding box [x_min, y_min, x_max, y_max]

# # Create pseudo video frames
# video_frames = create_pseudo_video(image, bounding_box)

# # Save video frames as a GIF file using imageio
# imageio.mimsave('pseudo_video.gif', [np.array(frame) for frame in video_frames], fps=10)
