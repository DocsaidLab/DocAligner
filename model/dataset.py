from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
from docsaidkit import INTER
from numpy import ndarray

DIR = D.get_curdir(__file__)

ds = D.load_json(DIR.parent / 'data' / 'indoor_dataset.json')

bg_dataset = []
for data in D.Tqdm(ds):
    img_path = D.Path('/data/Dataset') / data['img_path']
    if D.imread(img_path) is None:
        continue
    bg_dataset.append(img_path)


def check_boundary(img: Union[str, Path, np.ndarray], poly: np.ndarray):

    if isinstance(img, (str, Path)):
        img = D.imread(img)

    if img is None:
        return False

    poly = np.array(poly).reshape(-1, 2)
    if poly[:, 0].max() > img.shape[1] or poly[:, 1].max() > img.shape[0]:
        return False
    if poly[:, 0].min() < 0 or poly[:, 1].min() < 0:
        return False

    return True


class DefaultImageAug:

    def __init__(self, p=0.5):
        self.coarse_drop_aug = DT.CoarseDropout(
            max_holes=1,
            min_height=24,
            max_height=48,
            min_width=24,
            max_width=48,
            mask_fill_value=255,
            p=p
        )
        self.aug = A.Compose([

            DT.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=[-0.2, 0]
            ),

            A.OneOf([
                A.Spatter(mode='mud'),
                A.GaussNoise(),
                A.ISONoise(),
                A.MotionBlur(),
                A.Defocus(),
                A.GaussianBlur(blur_limit=(3, 11), p=0.5),
            ], p=p),

            A.OneOf([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
            ], p=p),

            A.OneOf([
                A.ColorJitter(),
                A.ChannelShuffle(),
                A.ChannelDropout(),
                A.RGBShift(),
            ])

        ], p=p, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> Any:
        mask = np.zeros_like(image)
        img, mask = self.coarse_drop_aug(image=image, mask=mask).values()
        background = bg_dataset[np.random.randint(len(bg_dataset))]
        background = D.imread(background)
        background = D.imresize(background, (image.shape[0], image.shape[1]))
        if mask.sum() > 0:
            img[mask > 0] = background[mask > 0]
        img, kps = self.aug(image=img, keypoints=keypoints).values()
        kps = D.order_points_clockwise(np.array(kps))
        return img, kps


class BaseDataset:

    def __init__(
        self,
        root: Union[str, Path] = None,
        image_size: Tuple[int, int] = None,
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        aug_func: Callable = None,
        aug_ratio: float = 0.0,
        **kwargs
    ) -> None:
        self.image_size = image_size
        self.interpolation = interpolation
        self.aug_ratio = aug_ratio
        self.root = DIR.parent.parent / \
            'dataset' if root is None else Path(root)
        self.dataset = self._build()
        self.aug_func = aug_func(p=aug_ratio) if aug_func is not None \
            else DefaultImageAug(p=aug_ratio)

    def __len__(self) -> int:
        return len(self.dataset)

    def _resize_poly(self, img: np.ndarray, poly: np.ndarray):
        h, w = img.shape[:2]
        img = D.imresize(img, self.image_size, self.interpolation)
        nh, nw = img.shape[:2]
        poly = D.Polygon(poly) \
            .normalize(w=w, h=h) \
            .denormalize(w=nw, h=nh) \
            .numpy()
        return img, poly

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        idx = idx % len(self.dataset)
        img_path, poly = self.dataset[idx]
        img = D.imread(img_path)

        if img is None:
            raise ValueError(f'Image is None: {img_path}')

        poly = np.array(poly)

        if self.image_size is not None:
            img, poly = self._resize_poly(img, poly)

        img, poly = self.aug_func(image=img, keypoints=poly)
        return img, poly

    def _build(self):
        raise NotImplementedError


class MIDV500Dataset(BaseDataset):

    def _build(self):

        if (fp := DIR.parent / 'data' / 'midv_dataset_500_cache.json').exists():
            return D.load_json(fp)

        ds = D.load_json(DIR.parent / 'data' / 'midv_dataset.json')
        dataset = []
        for data in D.Tqdm(ds.values()):
            for d in data:
                if D.Path(d['img_path']).parent.parent.parent.parent.stem == 'midv500':
                    img_path = self.root / d['img_path']
                    gt = D.load_json(self.root / d['gt_path'])['quad']
                    if check_boundary(img_path, gt):
                        dataset.append((str(img_path), gt))

        # Make a cache file
        if not fp.exists():
            D.dump_json(dataset, fp)

        return dataset


class MIDV2019Dataset(BaseDataset):

    def __init__(self, return_tensor: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.return_tensor = return_tensor

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'midv_dataset.json')
        dataset = []
        for data in D.Tqdm(ds.values()):
            for d in data:
                if D.Path(d['img_path']).parent.parent.parent.stem == 'midv2019':
                    img_path = self.root / d['img_path']
                    gt = D.load_json(self.root / d['gt_path'])['quad']
                    dataset.append((img_path, gt))
        return dataset

    def __getitem__(self, idx):
        if self.return_tensor:
            img, poly = super().__getitem__(idx)
            poly = D.Polygon(poly).normalize(
                w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
            img = np.transpose(img.astype('float32'), (2, 0, 1)) / 255.0
            return img, poly
        else:
            return super().__getitem__(idx)


class MIDV2020Dataset(BaseDataset):

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'midv2020_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            gt = D.load_json(self.root / data['gt_path'])['_via_img_metadata']
            for k, v in gt.items():
                if img_path.name in k:
                    idx = 0 if len(v['regions']) == 1 else 1
                    quad_x = v['regions'][idx]['shape_attributes']['all_points_x']
                    quad_y = v['regions'][idx]['shape_attributes']['all_points_y']
                    quad = np.array([quad_x, quad_y]).T
                    dataset.append((img_path, quad))
                    break
        return dataset


class CordDataset(BaseDataset):

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'cord_v0_train_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            gt = data['quad']
            dataset.append((img_path, gt))
        return dataset


class SmartDocDataset(BaseDataset):

    def __init__(
        self,
        mode: str = 'train',
        return_tensor: bool = False,
        train_ratio: float = 1,
        *args, **kwargs
    ) -> None:
        self.return_tensor = return_tensor
        self.mode = mode
        self.train_ratio = train_ratio
        super().__init__(*args, **kwargs)

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'smartdoc2015_dataset.json')
        dataset = []
        for key, data in D.Tqdm(ds.items()):
            if self.mode == 'train':
                data = data[:int(len(data) * self.train_ratio)]
            elif self.mode == 'val':
                data = data[int(len(data) * self.train_ratio):]
            for d in data:
                img_path = d['img_path']
                gt = D.load_json(d['gt_path'])
                dataset.append((img_path, gt, key))
        return dataset

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        # For Validation, SmartDocDataset has a special return value.
        # -> (img, poly, key)

        idx = idx % len(self.dataset)
        img_path, poly, key = self.dataset[idx]
        img = D.imread(img_path)

        if img is None:
            raise ValueError(f'Image is None: {img_path}')

        poly = np.array(poly)

        if self.image_size is not None:
            img, poly = self._resize_poly(img, poly)

        img, poly = self.aug_func(image=img, keypoints=poly)

        if self.return_tensor:
            poly = D.Polygon(poly).normalize(
                w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
            img = np.transpose(img.astype('float32'), (2, 0, 1)) / 255.0

        if self.mode == 'val':
            return img, poly, key

        return img, poly


class PrivateDataset(BaseDataset):

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'private_template' / 'gt.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = DIR.parent / 'data' / \
                'private_template' / data['path']
            gt = data['polygon']
            if len(gt) != 4:
                continue
            dataset.append((img_path, gt))
        return dataset


class SyncDataset(BaseDataset):

    def __init__(
        self,
        use_midv500: bool = True,
        use_midv2019: bool = True,
        use_midv2020: bool = True,
        use_cordv0: bool = True,
        use_smartdoc: bool = True,
        use_private: bool = True,
        length_of_dataset: int = 100000,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        kwargs.update({'aug_ratio': 0.0})

        if use_midv500:
            self.midv500 = MIDV500Dataset(**kwargs)
        if use_midv2019:
            self.midv2019 = MIDV2019Dataset(**kwargs)
        if use_midv2020:
            self.midv2020 = MIDV2020Dataset(**kwargs)
        if use_cordv0:
            self.cord = CordDataset(**kwargs)
        if use_smartdoc:
            self.smartdoc = SmartDocDataset(**kwargs)
        if use_private:
            self.private = PrivateDataset(**kwargs)

        self.pool = D.get_files(
            DIR.parent / 'data' / 'docpool', suffix=['.jpg'])
        self.length_of_dataset = length_of_dataset

    def __len__(self) -> int:
        return self.length_of_dataset

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'indoor_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            dataset.append((img_path, None))
        return dataset

    def _random_get_doc_image(self):
        # NOT USE midv2019 for getting doc image, because of the dataset has
        # a lot of incomplete document.

        tgts = ['pool']
        if hasattr(self, 'midv500'):
            tgts.append('midv500')
        if hasattr(self, 'midv2020'):
            tgts.append('midv2020')
        if hasattr(self, 'smartdoc'):
            tgts.append('smartdoc')

        tgt = np.random.choice(tgts)
        if tgt == 'midv500':
            return D.imwarp_quadrangle(
                *self.midv500[np.random.randint(len(self.midv500))])
        elif tgt == 'midv2020':
            return D.imwarp_quadrangle(
                *self.midv2020[np.random.randint(len(self.midv2020))])
        elif tgt == 'smartdoc':
            return D.imwarp_quadrangle(
                *self.smartdoc[np.random.randint(len(self.smartdoc))])
        else:
            return D.imread(self.pool[np.random.randint(len(self.pool))])

    def _generate_random_quadrant_points(self):
        # 0.3 -> Avoid the error order poins of the image.
        q1_point = np.random.uniform(low=[0.05, 0.05], high=[0.3, 0.3])
        q2_point = np.random.uniform(low=[0.6, 0.05], high=[0.95, 0.4])
        q3_point = np.random.uniform(low=[0.6, 0.6], high=[0.95, 0.95])
        q4_point = np.random.uniform(low=[0.05, 0.6], high=[0.4, 0.95])
        return np.array([q1_point, q2_point, q3_point, q4_point]).astype('float32')

    def _paste_doc_image(self, img, poly, doc_img):
        poly = D.Polygon(poly, normalized=True) \
            .denormalize(w=img.shape[1], h=img.shape[0]) \
            .numpy()

        poly_doc = np.array([
            [0, 0],
            [doc_img.shape[1], 0],
            [doc_img.shape[1], doc_img.shape[0]],
            [0, doc_img.shape[0]],
        ]).astype('float32')

        matrix = cv2.getPerspectiveTransform(poly_doc, poly)
        src_warp = cv2.warpPerspective(
            doc_img, matrix, (img.shape[1], img.shape[0]))
        sync_img = np.where(src_warp > 0, src_warp, img.copy())

        return sync_img, poly

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:

        tgts = []
        if hasattr(self, 'midv500'):
            tgts.append('midv500')
        if hasattr(self, 'midv2019'):
            tgts.append('midv2019')
        if hasattr(self, 'midv2020'):
            tgts.append('midv2020')
        if hasattr(self, 'smartdoc'):
            tgts.append('smartdoc')
        if hasattr(self, 'private'):
            tgts.append('private')

        if np.random.rand() < 0.5 or len(tgts) == 0:
            # Generate background image from indoor dataset.
            idx = np.random.randint(len(self.dataset))
            img_path, _ = self.dataset[idx]
            img = D.imread(img_path)
            if img is None:
                return self.__getitem__(idx)
            poly = self._generate_random_quadrant_points()
        else:
            # Generate background image from another dataset.
            tgt = np.random.choice(tgts)
            if tgt == 'midv500':
                img, poly = self.midv500[
                    np.random.randint(len(self.midv500))]
            elif tgt == 'midv2019':
                img, poly = self.midv2019[
                    np.random.randint(len(self.midv2019))]
            elif tgt == 'midv2020':
                img, poly = self.midv2020[
                    np.random.randint(len(self.midv2020))]
            elif tgt == 'smartdoc':
                img, poly = self.smartdoc[
                    np.random.randint(len(self.smartdoc))]
            elif tgt == 'private':
                img, poly = self.private[
                    np.random.randint(len(self.private))]

            poly = D.Polygon(poly).normalize(
                img.shape[1], img.shape[0]).numpy()

        doc_img = self._random_get_doc_image()
        sync_img, poly = self._paste_doc_image(img, poly, doc_img)

        if self.image_size is not None:
            sync_img, poly = self._resize_poly(sync_img, poly)

        sync_img, poly = self.aug_func(image=sync_img, keypoints=poly)
        return sync_img, poly


class BackgroundDataset(BaseDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        image_size = kwargs.get('image_size', None)
        self.random_resize_crop = A.RandomResizedCrop(
            height=image_size[0],
            width=image_size[1],
            p=1
        )

    def _build(self):
        return bg_dataset

    def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
        idx = np.random.randint(len(self.dataset))
        img = D.imread(self.dataset[idx])
        img = self.random_resize_crop(image=img)['image']
        poly = np.array([
            [-99, -99],
            [-99, -99],
            [-99, -99],
            [-99, -99]
        ]).astype('float32')
        return img, poly


class DocAlignerDataset:

    def __init__(
        self,
        root: Union[str, Path],
        image_size: Tuple[int, int] = (256, 256),
        edge_width: int = 3,
        downscale: int = 2,
        aug_ratio: float = 0.0,
        aug_func: Callable = None,
        fuse_dataset: List[Dict[str, Any]] = None,
        output_tensor: bool = True,
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        length_of_dataset: int = None,
        random_output: bool = False,
        random_nodoc_ratio: float = 0,
    ) -> None:
        self.root = root
        self.edge_width = edge_width
        self.downscale = downscale
        self.output_tensor = output_tensor
        self.random_output = random_output
        self.random_nodoc_ratio = random_nodoc_ratio
        self.background_dataset = BackgroundDataset(
            root=root, image_size=image_size)

        dataset = []
        n_dataset = []
        for ds in fuse_dataset:
            name = ds['name']
            options = ds['options']
            options.update({
                'root': root,
                'image_size': image_size,
                'interpolation': interpolation,
                'aug_func': aug_func,
                'aug_ratio': aug_ratio,
            })
            _ds = globals()[name](**options)
            dataset.append(_ds)
            n_dataset.append(len(_ds))

        self.dataset = dataset

        if random_output:
            length_of_dataset = 100000 if length_of_dataset is None else length_of_dataset
            self.length_of_dataset = [length_of_dataset]
        else:
            self.length_of_dataset = n_dataset

    def __len__(self) -> int:
        return sum(self.length_of_dataset)

    @staticmethod
    def _find_position_in_list(lst, target_sum):
        cumulative_sum = 0
        for i, value in enumerate(lst):
            if cumulative_sum + value >= target_sum:
                inner_index = target_sum - cumulative_sum
                return (i, inner_index)
            cumulative_sum += value
        return None

    def _gen_gaussian_point(self, img, poly, ksize: int = None):

        # 128 x 128 -> kszie ~= 7
        # 256 x 256 -> ksize ~= 15
        ksize = ksize if ksize is not None else (img.shape[0] // 17)
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getGaussianKernel(ksize, sigma=ksize//3)
        kernel = np.outer(kernel, kernel)

        # kernel is an ksize x ksize matrix, put gaussian kernel on the image.
        masks, mask_dils = [], []
        for p in poly:
            x, y = p
            x, y = int(x), int(y)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            if x <= 0 or x >= img.shape[1] or y <= 0 or y >= img.shape[0]:
                mask = D.imresize(mask, (img.shape[0] // self.downscale,
                                         img.shape[1] // self.downscale))
                mask_dil = D.imdilate(mask) > 0
                masks.append(mask)
                mask_dils.append(mask_dil)
                continue

            half_ksize = ksize // 2
            mask = D.pad(mask, pad_size=(half_ksize, half_ksize))

            x += half_ksize
            y += half_ksize

            # process exceed boundary
            min_y = y - half_ksize
            max_y = y + half_ksize + 1
            min_x = x - half_ksize
            max_x = x + half_ksize + 1

            # re-sclae the kernel into 0~255
            kernel = (kernel - kernel.min()) / \
                (kernel.max() - kernel.min()) * 255

            mask[min_y:max_y, min_x:max_x] = np.uint8(kernel)
            mask = mask[half_ksize:-half_ksize, half_ksize:-half_ksize]
            mask = D.imresize(mask, (img.shape[0] // self.downscale,
                                     img.shape[1] // self.downscale))
            mask_dil = D.imdilate(mask) > 0

            masks.append(mask)
            mask_dils.append(mask_dil)

        masks = np.stack(masks, axis=-1)
        mask_dils = np.stack(mask_dils, axis=-1)

        return masks, mask_dils

    def _gen_edge(self, img, poly):
        edge = cv2.fillPoly(np.zeros_like(img), [poly], color=(255, 255, 255))
        edge = D.imresize(
            edge, (edge.shape[0] // self.downscale, edge.shape[1] // self.downscale))
        edge = D.imgrandient(D.imbinarize(edge), ksize=self.edge_width)
        edge_mask = D.imdilate(edge, ksize=self.edge_width) > 0
        return edge, edge_mask

    def to_tensor(self, img, box, poly, edge, edge_mask, hmaps, hmaps_mask, has_obj):
        poly = D.Polygon(poly).normalize(
            w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
        box = D.Box(box).normalize(
            w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
        img = np.transpose(img.astype('float32'), (2, 0, 1)) / 255.0
        edge = edge.astype('float32') / 255.0
        edge_mask = edge_mask.astype('float32')
        hmaps = np.transpose(hmaps.astype('float32'), (2, 0, 1)) / 255.0
        hmaps_mask = np.transpose(hmaps_mask.astype('float32'), (2, 0, 1))
        has_obj = np.array([has_obj]).astype('float32')
        return img, box, poly, edge, edge_mask, hmaps, hmaps_mask, has_obj

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.rand() < self.random_nodoc_ratio:
            # Generate background image from indoor dataset.
            img, poly = self.background_dataset[idx]
            has_obj = False
        else:
            if self.random_output:
                d_idx = np.random.randint(len(self.dataset))
                f_idx = np.random.randint(len(self.dataset[d_idx]))
            else:
                d_idx, f_idx = self._find_position_in_list(
                    self.length_of_dataset, idx)
            img, poly = self.dataset[d_idx][f_idx]
            has_obj = True

        hmaps, hmaps_mask = self._gen_gaussian_point(img, poly.astype('int32'))
        edge, edge_mask = self._gen_edge(img, poly.astype('int32'))

        # Polygon -> Bounding Box
        box = D.Polygon(poly).to_box(box_mode='XYWH').numpy()

        if self.output_tensor:
            return self.to_tensor(img, box, poly, edge, edge_mask, hmaps, hmaps_mask, has_obj)

        return img, box, poly, edge, edge_mask, hmaps, hmaps_mask, has_obj
