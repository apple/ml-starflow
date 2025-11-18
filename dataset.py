#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import io
import os
import csv
import json
import random
import torch
import numpy as np
import math
import time
import contextlib
from typing import Optional, Union
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import default_collate, get_worker_info
import tarfile
import tqdm
import gc
import threading
import psutil
import tempfile
import decord
from decord import VideoReader
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from misc import print, xprint
from misc.condition_utils import get_camera_condition, get_point_condition, get_wind_condition

# Initialize multiprocessing manager
manager = torch.multiprocessing.Manager()

# ==== helpers ==== #

@contextlib.contextmanager
def ram_temp_file(data, suffix=".mp4"):
    available_ram = psutil.virtual_memory().available
    video_size = len(data)
    
    # Use RAM if available, otherwise fall back to disk
    if video_size < available_ram - (500 * 1024 * 1024):
        temp_dir = "/dev/shm"  # RAM disk
    else:
        temp_dir = None  # Default system temp (disk)

    with tempfile.NamedTemporaryFile(dir=temp_dir, suffix=suffix, delete=True) as temp_file:
        temp_file.write(data)
        temp_file.flush()
        yield temp_file.name


def _nearest_multiple(x: float, base: int = 8) -> int:
    """Round x to the nearest multiple of `base`."""
    return int(round(x / base)) * base


def aspect_ratio_to_image_size(target_size, R, multiple=8):
    if R is None:
        return target_size, target_size
    if isinstance(R, str):
        rw, rh = map(int, R.split(':'))
        R = rw / rh
    area  = target_size ** 2
    out_h = _nearest_multiple(math.sqrt(area / R), multiple)
    out_w = _nearest_multiple(math.sqrt(area * R), multiple)
    return out_h, out_w


def read_tsv(filename):
    # Open the TSV file for reading
    with open(filename, 'r', newline='') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        rows = []
        while True:
            try:
                r = next(reader)
                rows.append(r)
            except csv.Error as e:
                print(f'{e}')
            except StopIteration:
                break
        return rows


def sample_clip(
    video_path: str,
    num_frames: int = 8,
    out_fps: Optional[float] = None,      # ← pass an fps here
):
    vr       = VideoReader(video_path)
    src_fps  = vr.get_avg_fps()        # native fps
    total    = len(vr)

    if out_fps is None or out_fps >= src_fps:
        step = 1                       # keep native rate or up-sample later
    else:
        target_duration = (num_frames - 1) / out_fps  # duration in seconds
        frame_span = target_duration * src_fps   # frames needed for this duration
        step = max(frame_span / (num_frames - 1), 1)

    max_start = total - step * (num_frames - 1)
    if max_start <= 1:  # video too short for requested clip
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        return vr.get_batch(indices.tolist()), indices

    max_start = int(np.floor(max_start - 1))
    start  = random.randint(0, max_start) if max_start > 0 else 0
    idxs   = [int(np.round(start + i * step)) for i in range(num_frames)]
    return vr.get_batch(idxs), idxs


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            print('Another Loop over the dataset', flush=True)
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


class DataLoaderWrapper(InfiniteDataLoader):
    def __iter__(self):
        return IterWrapper(super().__iter__())


class IterWrapper:
    def __init__(self, obj):
        self.obj = obj

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()

    def next(self):
        return next(self.obj)


# ==== Dataset Implementation, Load your own data ==== #

class ImageTarDataset(Dataset):
    def __init__(self, dataset_tsv, image_size, temporal_size=None, rank=0,  world_size=1, 
                 use_image_bucket=False, multiple=8, no_flip=False, edit=False):
        all_lines = []
                
        # get all data lines
        self.buckets = {}
        self.weights = {}
        self.image_buckets = defaultdict(lambda: 0)
        self.image_buckets['1:1'] = 0  # default bucket

        skipped = 0
        for line in tqdm.tqdm(read_tsv(dataset_tsv)[1:]):
            tsv_file = line[0]
            bucket   = line[1] if len(line) > 1 else 'mlx'
            caption  = line[2] if len(line) > 2 else 'caption'
            weights  = float(line[3] if len(line) > 3 else "1")
            all_data = read_tsv(tsv_file)
            all_maps = {all_data[0][i]: i for i in range(len(all_data[0]))}
            self.weights[all_data[1][0]] = weights
            for line in all_data[1:]:
                try:
                    if 'width' in all_maps:  # filter too small images
                        width, height = int(line[all_maps['width']]), int(line[all_maps['height']])
                        if width * height < (image_size * image_size) / 2:  # if image is smaller than half size of the target size
                            skipped += 1; continue

                    if caption != 'folder':  # input caption has higher priority
                        captions  = caption.split('|')[0].split(':') 
                        operation = caption.split('|')[1] if len(caption.split('|')) > 1 else "none"
                        caption_line = ([line[all_maps[c]] for c in captions], operation)
                    else:
                        caption_line = (line[all_maps['file']].split('/')[-2], "none")  # use folder name as caption
                    
                    items = {'tar': line[all_maps['tar']], 'file': line[all_maps['file']], 'caption': caption_line,
                             'image_bucket': line[all_maps['image_bucket']] if 'image_bucket' in all_maps else "1:1"}
                    
                    if "camera_file" in all_maps: # dl3dv data
                        items["camera_file"] = line[all_maps["camera_file"]]
                    
                    if "force_caption" in all_maps: # force dataset
                        items["force_caption"] = line[all_maps["force_caption"]]
                        if "wind_speed" in all_maps: # wind force
                            items["wind_speed"] = line[all_maps["wind_speed"]]
                            items["wind_angle"] = line[all_maps["wind_angle"]]
                        elif "force" in all_maps: # point-wise
                            items["force"] = line[all_maps["force"]]
                            items["angle"] = line[all_maps["angle"]]
                            items["coordx"] = line[all_maps["coordx"]]
                            items["coordy"] = line[all_maps["coordy"]]
                        
                    if edit:
                        if line[all_maps['visual_file']] != 'none': continue  # TODO: for now, we only support one image, no visual clue    
                        items['edit_instruction'] = line[all_maps['edit_instruction']]
                        items['edited_file'] = line[all_maps['edited_file']]
                    all_lines.append(items)
                    
                except Exception as e:
                    skipped += 1; continue
                
                image_bucket = all_lines[-1]['image_bucket']
                self.image_buckets[image_bucket] += 1
                if all_lines[-1]['tar'] not in self.buckets:
                    self.buckets[all_lines[-1]['tar']] = bucket
        
        if "force_caption" in all_lines[0]:            
            wind_forces = [l["wind_speed"] for l in all_lines] if "wind_speed" in all_lines[0] else [l["force"] for l in all_lines]
            self.min_wind_force = min(wind_forces)
            self.max_wind_force = max(wind_forces)

        self.use_image_bucket = use_image_bucket
        self.all_lines = all_lines[rank:][::world_size]   # all lines is sorted by tar file
        self.num_samples_per_rank = None
        self.image_size = image_size
        self.multiple = multiple
        self.temporal_size = tuple(map(int, temporal_size.split(':'))) if isinstance(temporal_size, str) else None
        self.edit_mode = edit
        
        def center_crop_resize(img, ratio="1:1", target_size: int = 256, multiple: int = 8):
            """
            1. Center crop `img` to the largest window with aspect ratio = ratio.
            2. Resize so  HxW ≈ target_size²  (each side a multiple of `multiple`).

            Args
            ----
            img         : PIL Image or torch tensor (CHW/HWC)
            ratio       : "3:2", (3,2), "1:1", etc.
            target_size : reference side length (area = target_size²)
            multiple    : force each output side to be a multiple of this number
            """
            # --- parse ratio ----------------------------------------------------------
            if isinstance(ratio, str):
                rw, rh = map(int, ratio.split(':'))
            else:                                 # already a tuple/list
                rw, rh = ratio
            R = rw / rh                           # width / height

            # --- crop to that aspect ratio -------------------------------------------
            w, h = img.size if hasattr(img, "size") else (img.shape[-1], img.shape[-2])
            if w / h > R:                         # image too wide → trim width
                crop_h, crop_w = h, int(round(h * R))
            else:                                 # image too tall → trim height
                crop_w, crop_h = w, int(round(w / R))
            img = transforms.functional.center_crop(img, (crop_h, crop_w))

            # --- compute output dimensions -------------------------------------------
            area  = target_size ** 2
            out_h = _nearest_multiple(math.sqrt(area / R), multiple)
            out_w = _nearest_multiple(math.sqrt(area * R), multiple)

            # --- resize & return ------------------------------------------------------
            return transforms.functional.resize(img, (out_h, out_w), antialias=True)
        
        self.transforms = {}
        self.size_bucket_maps = {}
        self.bucket_size_maps = {}
        for bucket in self.image_buckets:
            trans = [transforms.Lambda(lambda img, r=bucket: center_crop_resize(img, ratio=r, target_size=image_size, multiple=multiple))]
            if not no_flip:
                trans.append(transforms.RandomHorizontalFlip())
            trans.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.transforms[bucket] = transforms.Compose(trans)

            w, h = map(int, bucket.split(':'))
            out_h, out_w = aspect_ratio_to_image_size(image_size, w / h, multiple=multiple)
            self.size_bucket_maps[(out_h, out_w)] = bucket
            self.bucket_size_maps[bucket] = (out_h, out_w)
            
        self.transform = self.transforms['1:1']  # default transform
        print(f"Rank0 -- Loading {len(self.all_lines)} lines of data | {skipped} lines are skipped due to size or error")

    def __len__(self):
        if self.num_samples_per_rank is not None:
            return self.num_samples_per_rank
        return len(self.all_lines)
    
    def __getitem__(self, idx):
        image_item = self.all_lines[idx]
        tar_file = image_item['tar']
        img_file = image_item['file']
        img_bucket = image_item['image_bucket']
        try:
            with tarfile.open(tar_file, mode='r') as tar:
                img = self._read_image(tar, img_file, img_bucket)
                H0, W0 = img.size
                scale  = self.image_size / min(H0, W0)
                state  = np.array([scale, H0, W0])
        except Exception as e:
            print(f'Reading data error {e}')
        sample = image_item.copy()
        sample.update(image=img, state=state)
        return sample
    
    def _read_image(self, tar, img_file, img_bucket):
        
        def _transform(img):
            if not self.use_image_bucket:
                return self.transform(img)
            else:
                return self.transforms[img_bucket](img)

        x_shape = aspect_ratio_to_image_size(self.image_size, img_bucket, multiple=self.multiple)
        if self.temporal_size is not None:  # read video
            num_frames, out_fps = self.temporal_size[0], self.temporal_size[1:]
            if len(out_fps) == 1:
                out_fps = out_fps[0]
            else:
                out_fps = random.choice(out_fps)  # randomly choose one fps from the list            
            assert img_file.endswith('.mp4'), "Only support mp4 video for now"   
            try:
                with tar.extractfile(img_file) as video_data:
                    with ram_temp_file(video_data.read()) as tmp_path:
                        frames, frame_inds = sample_clip(tmp_path, num_frames=num_frames, out_fps=out_fps)
                        frames = frames.asnumpy()    
            except Exception as e:
                print(f'Reading data error {e} {img_file}')
                frames = np.zeros((num_frames, x_shape[0], x_shape[1], 3), dtype=np.uint8)
            return torch.stack([_transform(Image.fromarray(frame)) for frame in frames]), out_fps, frame_inds
        
        try:           
            original_img = Image.open(tar.extractfile(img_file)).convert('RGB')
        except Exception as e:
            print(f'Reading data error {e} {img_file}')
            original_img = Image.new('RGB', (x_shape[0], x_shape[1]), (0, 0, 0))
        return _transform(original_img), 0, None

    def collate_fn(self, batch):
        batch = default_collate(batch)
        return batch

    def get_batch_modes(self, x):
        x_aspect   = self.size_bucket_maps.get(x.size()[-2:], "1:1")
        video_mode = self.temporal_size is not None
        return x_aspect, video_mode


class OnlineImageTarDataset(ImageTarDataset):
    max_retry_n = 20
    max_read = 4096
    tar_keys_lock = manager.Lock() if manager is not None else None
    
    def __init__(self, dataset_tsv, image_size, batch_size=None, **kwargs):
        super().__init__(dataset_tsv, image_size, **kwargs)
        
        self.tar_lists = defaultdict(lambda: [])
        self.tar_image_buckets = defaultdict(lambda: defaultdict(lambda: 0))
        for i, line in enumerate(self.all_lines):
            tar_file = line['tar']
            image_bucket = line['image_bucket']
            self.tar_lists[tar_file] += [i]
            self.tar_image_buckets[tar_file][image_bucket] += 1
        self.reset_tar_keys = []
        for key in self.tar_lists.keys():
            repeat = int(self.weights.get(key, 1))
            self.reset_tar_keys.extend([key] * repeat)
        self.tar_keys = manager.list(self.reset_tar_keys) if manager is not None else list(self.reset_tar_keys)
        
        # Use more workers for better prefetching, but limit to reasonable number
        self.worker_executors = {}
        self.worker_caches = {}  # each entry: {active:{tar,key,cnt,inner_idx}, prefetch:{future,key}} 
        self.worker_caches_lock = threading.Lock()  # Protect worker_caches access
        self.shuffle_everything()
        if self.use_image_bucket:
            assert batch_size, "batch_size should be set when use_image_bucket is True"
        self.batch_size = batch_size
        if self.temporal_size is not None:
            assert self.temporal_size[0] > 1, "temporal_size should be greater than 1 for video data" 
            self.max_read = 512
    
    def cleanup_worker_cache(self, wid):
        """Clean up worker cache entry and associated resources"""
        with self.worker_caches_lock:
            if wid in self.worker_caches:
                cache_entry = self.worker_caches[wid]
                # Cancel prefetch future if still running
                if 'prefetch' in cache_entry and hasattr(cache_entry['prefetch'], 'cancel'):
                    cache_entry['prefetch'].cancel()
                
                if cache_entry.get('tar') is not None:
                    tar = cache_entry['tar']
                    self._close_tar(tar)
                    cache_entry['tar'] = None
                # Remove the entire cache entry
                del self.worker_caches[wid]
                gc.collect()
    
    def _s3(self):
        raise NotImplementedError("Please implement your own _s3() method to return a boto3 session/client")
    
    def shuffle_everything(self):
        for key in tqdm.tqdm(self.tar_keys):
            random.shuffle(self.tar_lists[key])
        random.shuffle(self.tar_keys)
        print("shuffle everything done!")
    
    def download_tar(self, prefetch=True, wid=None):
        i = 0
        file_stream = None
        tar_file = None
        download = f'prefetch {wid}' if prefetch else 'just download'
        while True:
            if i % self.max_retry_n == 0:  # retry a different tar file
                tar_file = self._get_next_key()  # get the next tar file key
            file_stream = None
            try:
                file_stream = io.BytesIO()
                self._s3().download_fileobj(self.buckets[tar_file], tar_file, file_stream)  # hard-coded
                file_stream.seek(0)
                tar = tarfile.open(fileobj=file_stream, mode='r')
                # Store the file_stream reference so it can be closed later
                tar._file_stream = file_stream
                xprint(f'[INFO] {download} tar file: {tar_file}')
                return tar, tar_file
            except Exception as e:
                xprint(f'[ERROR] {download} tar file {tar_file} failed: {e}')
                i += 1
                if file_stream:
                    file_stream.close()
                    file_stream = None
                time.sleep(min(i * 0.1, 5))  # Exponential backoff with cap
        
    def _get_next_key(self):
        with self.tar_keys_lock:
            if not self.tar_keys or len(self.tar_keys) == 0:
                xprint(f'[WARN] all dataset exhausted... this should not happen usually')
                self.tar_keys.extend(list(self.reset_tar_keys))  # reset
                random.shuffle(self.tar_keys)
            return self.tar_keys.pop(0)  # remove and return the first key
    
    def _start_prefetch(self, wid):
        """Start prefetching the next tar file for the worker"""
        # Create executor per worker process if it doesn't exist
        if wid not in self.worker_executors:
            self.worker_executors[wid] = ThreadPoolExecutor(max_workers=1)
        future = self.worker_executors[wid].submit(self.download_tar, prefetch=True, wid=wid)  # download tar file in a separate thread
        self.worker_caches[wid]['prefetch'] = future
    
    def _close_tar(self, tar):
        # Properly close both tar and underlying file stream
        if hasattr(tar, '_file_stream') and tar._file_stream:
            tar._file_stream.close()
            tar._file_stream = None
        tar.close()
        del tar
        gc.collect()
    
    def __getitem__(self, idx):        
        try:
            wid = get_worker_info().id
        except Exception as e:
            wid = -1
        
        # ─── first time this worker is used ─── #
        if wid not in self.worker_caches:
            tar, key = self.download_tar(prefetch=False)  # download tar file
            with self.worker_caches_lock:
                self.worker_caches[wid] = dict(
                    active=dict(tar=tar, key=key, cnt=0, inner_idx=0),  # active cache
                )
                self._start_prefetch(wid)  # start prefetching the next tar file
        
        cache = self.worker_caches[wid]
        active = cache['active']
        tar = active['tar']
        key = active['key']
        cnt = active['cnt']
        inner_idx = active['inner_idx']
        
        # handle image bucketting
        if self.use_image_bucket:
            if inner_idx % self.batch_size == 0:
                # sample based on local tar file statistics in case some dataset only has one image bucket
                tar_buckets = self.tar_image_buckets[key]
                target_image_bucket = random.choices(
                    list(tar_buckets.keys()), weights=list(tar_buckets.values()), k=1)[0]
                self.worker_caches[wid]['target_image_bucket'] = target_image_bucket
            
            # scan the list to find the nearest target image bucket
            target_image_bucket, t_cnt = self.worker_caches[wid]['target_image_bucket'], cnt
            while self.all_lines[self.tar_lists[key][t_cnt]]['image_bucket'] != target_image_bucket:
                t_cnt += 1
                if t_cnt >= len(self.tar_lists[key]): t_cnt = 0
            # sawp the image location
            if cnt != t_cnt:
                self.tar_lists[key][cnt], self.tar_lists[key][t_cnt] = self.tar_lists[key][t_cnt], self.tar_lists[key][cnt]
                    
        img_id = self.tar_lists[key][cnt]
        image_item = self.all_lines[img_id]    
        sample = {key: image_item[key] for key in image_item}
        image, fps, frame_inds = self._read_image(tar, image_item['file'], image_item['image_bucket'])
        sample.update(image=image, fps=fps, local_idx=img_id, inner_idx=inner_idx)
        if self.edit_mode:
            image, fps, _ = self._read_image(tar, image_item['edited_file'], image_item['image_bucket'])
            sample.update(edited_image=image, fps=fps, edit_instruction=image_item['edit_instruction'])
            
        if "camera_file" in image_item: # dl3dv data
            sample["condition"] = get_camera_condition(tar, image_item["camera_file"], width=image.shape[3], height=image.shape[2], factor=self.multiple, frame_inds=frame_inds)
        
        if "force_caption" in image_item: # force dataset
            if "wind_speed" in image_item: # wind force
                sample["condition"] = get_wind_condition(image_item["wind_speed"], image_item["wind_angle"], min_force=self.min_wind_force, max_force=self.max_wind_force, num_frames=image.shape[1], width=image.shape[3], height=image.shape[2]) 
            elif "force" in image_item: # point-wise
                sample["condition"] = get_point_condition(image_item["force"], image_item["angle"], image_item["coordx"], image_item["coordy"], min_force=self.min_wind_force, max_force=self.max_wind_force, num_frames=image.shape[1], width=image.shape[3], height=image.shape[2]) 
        
        # update cnt
        cnt, inner_idx = cnt + 1, inner_idx + 1
        if (cnt == len(self.tar_lists[key])) or (cnt == self.max_read):
            # -- active tar finished, switch to prefetched tar -- #
            self._close_tar(tar)  # close the current tar file
            
            try:
                # Wait for prefetch with timeout
                new_tar, new_key = cache['prefetch'].result()  # 5 minute timeout
            except Exception as e:
                xprint(f'[WARN] Prefetch failed, downloading new tar synchronously: {e}')
                new_tar, new_key = self.download_tar(prefetch=False)
                
            cache['active'] = dict(tar=new_tar, key=new_key, cnt=0, inner_idx=inner_idx)  # update active cache
            
            # shuffle the image list
            random.shuffle(self.tar_lists[key])  # shuffle the list
            with self.tar_keys_lock:
                self.tar_keys.append(key)  # return the key to the list so other workers can use it
            
            self._start_prefetch(wid)  # start prefetching the next tar file
        else:
            cache['active']['cnt'] = cnt
        
        # always update inner_idx (IMPORTANT)
        cache['active']['inner_idx'] = inner_idx
        return sample


class OnlineImageCaptionDataset(OnlineImageTarDataset):
    def __getitem__(self, idx):   
        sample = super().__getitem__(idx)
        captions, caption_op = sample['caption']
        if caption_op == 'none':
            sample['caption'] = captions[0] if isinstance(captions, list) else captions
        elif ':' in caption_op:
            sample['caption'] = random.choices(captions, weights=[float(a) for a in caption_op.split(':')])[0]
        else:
            raise NotImplementedError(f"Unknown caption operation: {caption_op}") 
        return sample

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        image = batch['image']
        caption = batch['caption']
        if self.edit_mode:
            image = torch.cat([image, batch['edited_image']], dim=0)
            caption.extend(batch['edit_instruction'])
        
        meta = {key: batch[key] for key in batch if key not in 
                ['image', 'caption', 'edited_image', 'edit_instruction']}
        return image, caption, meta


# ==== Dummy Dataset Implementation for Open Source Release ====

class DummyImageCaptionDataset(Dataset):
    """
    Dummy dataset that generates synthetic image-caption pairs for training/testing.
    Supports mixed aspect ratios and batch-wise aspect ratio consistency.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 256,
        temporal_size: Optional[str] = None,
        use_image_bucket: bool = False,
        batch_size: Optional[int] = None,
        multiple: int = 8,
        no_flip: bool = False,
        edit: bool = False
    ):
        """
        Args:
            num_samples: Number of samples in the dataset
            image_size: Base image size for generation
            temporal_size: Video size specification (e.g., "16:8" for frames:fps)
            use_image_bucket: Whether to use aspect ratio bucketing
            batch_size: Batch size for bucketing (required if use_image_bucket=True)
            multiple: Multiple for dimension rounding
            no_flip: Whether to disable horizontal flipping
            edit: Whether this is an editing dataset
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.temporal_size = temporal_size
        self.use_image_bucket = use_image_bucket
        self.batch_size = batch_size
        self.multiple = multiple
        self.no_flip = no_flip
        self.edit_mode = edit

        # Parse video parameters
        self.is_video = temporal_size is not None
        if self.is_video:
            frames, fps = map(int, temporal_size.split(':'))
            self.num_frames = frames
            self.fps = fps
        else:
            self.num_frames = 1
            self.fps = None

        # Aspect ratios for mixed aspect ratio training
        self.aspect_ratios = [
            "1:1", "2:3", "3:2", "16:9", "9:16",
            "4:5", "5:4", "21:9", "9:21"
        ] if use_image_bucket else ["1:1"]

        # Generate image buckets for aspect ratios
        self.image_buckets = {}
        for i, ar in enumerate(self.aspect_ratios):
            h, w = aspect_ratio_to_image_size(image_size, ar, multiple)
            self.image_buckets[ar] = (h, w, ar)

        # Sample captions for dummy data
        self.sample_captions = [
            "A beautiful landscape with mountains and trees",
            "A cute cat sitting on a wooden table",
            "A modern city skyline at sunset",
            "A vintage car parked on a street",
            "A delicious meal on a white plate",
            "A person walking in a park",
            "A colorful flower garden in bloom",
            "A cozy living room with furniture",
            "A stormy ocean with large waves",
            "A peaceful forest path in autumn",
            "A group of friends laughing together",
            "A majestic eagle flying in the sky",
            "A busy marketplace with vendors",
            "A snow-covered mountain peak",
            "A child playing with toys",
            "A romantic candlelit dinner",
            "A train traveling through countryside",
            "A lighthouse on a rocky coast",
            "A field of sunflowers under blue sky",
            "A family having a picnic outdoors"
        ]

        # Create transform pipeline
        def center_crop_resize(img, ratio="1:1", target_size: int = 256, multiple: int = 8):
            """
            1. Center crop `img` to the largest window with aspect ratio = ratio.
            2. Resize so  HxW ≈ target_size²  (each side a multiple of `multiple`).

            Args
            ----
            img         : PIL Image or torch tensor (CHW/HWC)
            ratio       : "3:2", (3,2), "1:1", etc.
            target_size : reference side length (area = target_size²)
            multiple    : force each output side to be a multiple of this number
            """
            # --- parse ratio ----------------------------------------------------------
            if isinstance(ratio, str):
                rw, rh = map(int, ratio.split(':'))
            else:                                 # already a tuple/list
                rw, rh = ratio
            R = rw / rh                           # width / height

            # --- crop to that aspect ratio -------------------------------------------
            w, h = img.size if hasattr(img, "size") else (img.shape[-1], img.shape[-2])
            if w / h > R:                         # image too wide → trim width
                crop_h, crop_w = h, int(round(h * R))
            else:                                 # image too tall → trim height
                crop_w, crop_h = w, int(round(w / R))
            img = transforms.functional.center_crop(img, (crop_h, crop_w))

            # --- compute output dimensions -------------------------------------------
            area  = target_size ** 2
            out_h = _nearest_multiple(math.sqrt(area / R), multiple)
            out_w = _nearest_multiple(math.sqrt(area * R), multiple)

            # --- resize & return ------------------------------------------------------
            return transforms.functional.resize(img, (out_h, out_w), antialias=True)
        
        self.transforms = {}
        self.size_bucket_maps = {}
        self.bucket_size_maps = {}
        for bucket in self.image_buckets:
            trans = [transforms.Lambda(lambda img, r=bucket: center_crop_resize(img, ratio=r, target_size=image_size, multiple=multiple))]
            if not no_flip:
                trans.append(transforms.RandomHorizontalFlip())
            trans.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            self.transforms[bucket] = transforms.Compose(trans)

            w, h = map(int, bucket.split(':'))
            out_h, out_w = aspect_ratio_to_image_size(image_size, w / h, multiple=multiple)
            self.size_bucket_maps[(out_h, out_w)] = bucket
            self.bucket_size_maps[bucket] = (out_h, out_w)
            
        self.transform = self.transforms['1:1']  # default transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample from the dataset."""
        # Choose aspect ratio
        if self.use_image_bucket:
            bucket_name = random.choice(list(self.image_buckets.keys()))
            h, w, aspect_ratio = self.image_buckets[bucket_name]
        else:
            h, w, aspect_ratio = self.image_size, self.image_size, "1:1"
            bucket_name = aspect_ratio

        # Generate dummy image
        if self.is_video:
            # Generate video tensor (T, C, H, W)
            image = torch.randn(self.num_frames, 3, h, w)
            # Normalize to [-1, 1] range
            image = torch.tanh(image)
        else:
            # Generate RGB image
            image = Image.new('RGB', (w, h), color=(
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            ))

            # Add some random patterns for variety
            if random.random() > 0.5:
                # Add gradient
                pixels = []
                for y in range(h):
                    for x in range(w):
                        r = int(255 * x / w)
                        g = int(255 * y / h)
                        b = int(255 * (x + y) / (w + h))
                        pixels.append((r, g, b))
                image.putdata(pixels)

            image = self.transform(image)

        # Generate caption
        caption = random.choice(self.sample_captions)

        # Add some variation to captions
        if random.random() > 0.7:
            adjectives = ["beautiful", "stunning", "amazing", "incredible", "magnificent"]
            caption = f"{random.choice(adjectives)} {caption.lower()}"

        sample = {
            'image': image,
            'caption': caption,
            'image_bucket': bucket_name,
            'aspect_ratio': aspect_ratio,
            'idx': idx
        }

        # Add video-specific metadata
        if self.is_video:
            sample.update({
                'num_frames': self.num_frames,
                'fps': self.fps,
                'temporal_size': self.temporal_size
            })

        # Add editing data if needed
        if self.edit_mode:
            # Generate slightly modified image for editing tasks
            edited_image = image + torch.randn_like(image) * 0.1
            edited_image = torch.clamp(edited_image, -1, 1)
            sample.update({
                'edited_image': edited_image,
                'edit_instruction': f"Edit this image to make it more {random.choice(['colorful', 'bright', 'artistic', 'realistic'])}"
            })

        return sample

    def collate_fn(self, batch: list) -> tuple:
        """Collate function for batching samples."""
        # Group by aspect ratio if using image buckets
        if self.use_image_bucket:
            # Sort batch by image bucket for consistency
            batch = sorted(batch, key=lambda x: x['image_bucket'])

        # Standard collation
        collated = {}
        images = torch.stack([item['image'] for item in batch], dim=0)
        captions = [item['caption'] for item in batch]

        # Collect metadata
        for key in ['image_bucket', 'aspect_ratio', 'idx']:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]

        # Handle video metadata
        if self.is_video:
            for key in ['num_frames', 'fps', 'temporal_size']:
                if key in batch[0]:
                    collated[key] = [item[key] for item in batch]

        # Handle editing data
        if self.edit_mode and 'edited_image' in batch[0]:
            edited_images = torch.stack([item['edited_image'] for item in batch], dim=0)
            collated['edited_image'] = edited_images
            collated['edit_instruction'] = [item['edit_instruction'] for item in batch]

        return images, captions, collated

    def get_batch_modes(self, x):
        x_aspect   = self.size_bucket_maps.get(x.size()[-2:], "1:1")
        video_mode = self.temporal_size is not None
        return x_aspect, video_mode


class DummyDataLoaderWrapper:
    """
    Wrapper that mimics the DataLoaderWrapper functionality.
    Provides infinite iteration over the dataset.
    """

    def __init__(self, dataset, batch_size=1, num_workers=0, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            drop_last=True,
            **kwargs
        )
        self.iterator = None
        self.secondary_loader = None

    def __iter__(self):
        """Infinite iteration over the dataset."""
        while True:
            if self.iterator is None:
                self.iterator = iter(self.dataloader)
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                yield next(self.iterator)

    def __len__(self):
        return len(self.dataloader)


def create_dummy_dataloader(
    dataset_name: str,
    img_size: int,
    vid_size: Optional[str] = None,
    batch_size: int = 16,
    use_mixed_aspect: bool = False,
    multiple: int = 8,
    num_samples: int = 10000,
    infinite: bool = False
) -> Union[DataLoader, DummyDataLoaderWrapper]:
    """
    Create a dummy dataloader that mimics the original functionality.

    Args:
        dataset_name: Name of the dataset (used for deterministic seeding)
        img_size: Base image size
        vid_size: Video specification (e.g., "16:8")
        batch_size: Batch size
        use_mixed_aspect: Whether to use mixed aspect ratio training
        multiple: Multiple for dimension rounding
        num_samples: Number of samples in the dataset
        infinite: Whether to create infinite dataloader

    Returns:
        DataLoader or DummyDataLoaderWrapper
    """
    # Set seed based on dataset name for reproducibility
    seed = hash(dataset_name) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)

    # Create dataset
    dataset = DummyImageCaptionDataset(
        num_samples=num_samples,
        image_size=img_size,
        temporal_size=vid_size,
        use_image_bucket=use_mixed_aspect,
        batch_size=batch_size,
        multiple=multiple,
        edit='edit' in dataset_name.lower()
    )

    # Set dataset attributes expected by training code
    dataset.total_num_samples = num_samples
    dataset.num_samples_per_rank = num_samples

    # Create dataloader
    if infinite:
        return DummyDataLoaderWrapper(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True
        )
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            collate_fn=dataset.collate_fn,
            persistent_workers=True
        )