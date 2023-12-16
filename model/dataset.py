from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import albumentations as A
import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import numpy as np
from docsaidkit import INTER

DIR = D.get_curdir(__file__)


class DefaultImageAug:

    def __init__(self, p=0.5):
        self.coarse_drop_aug = DT.CoarseDropout(
            max_holes=1, max_height=64, max_width=64, p=p)
        self.aug = A.Compose([
            DT.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=[-0.6, 0.2]
            ),
            A.MotionBlur(),
            A.GaussNoise(),
            A.ColorJitter(),
            A.ChannelShuffle(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.Perspective(),
            A.GaussianBlur(blur_limit=(7, 11), p=0.5),
        ], p=p, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __call__(self, image: np.ndarray, keypoints: np.ndarray) -> Any:
        img = self.coarse_drop_aug(image=image)['image']
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
        ds = D.load_json(DIR.parent / 'data' / 'midv_dataset.json')
        dataset = []
        for data in D.Tqdm(ds.values()):
            for d in data:
                if D.Path(d['img_path']).parent.parent.parent.parent.stem == 'midv500':
                    img_path = self.root / d['img_path']
                    gt = D.load_json(self.root / d['gt_path'])['quad']
                    dataset.append((img_path, gt))
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

    def __init__(self, mode: str = 'train', return_tensor: bool = False, *args, **kwargs) -> None:
        self.return_tensor = return_tensor
        self.mode = mode
        super().__init__(*args, **kwargs)

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'smartdoc2015_dataset.json')
        dataset = []
        for key, data in D.Tqdm(ds.items()):
            if self.mode == 'train':
                data = data[:int(len(data) * 0.2)]
            elif self.mode == 'val':
                data = data[int(len(data) * 0.2):]
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
        use_private: bool = False,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        kwargs.update({'aug_ratio': 0.0})

        target_dataset = []
        if use_midv500:
            self.midv500 = MIDV500Dataset(**kwargs)
            target_dataset.append('midv500')
        if use_midv2019:
            self.midv2019 = MIDV2019Dataset(**kwargs)
            target_dataset.append('midv2019')
        if use_midv2020:
            self.midv2020 = MIDV2020Dataset(**kwargs)
            target_dataset.append('midv2020')
        if use_cordv0:
            self.cord = CordDataset(**kwargs)
            target_dataset.append('cord')
        if use_smartdoc:
            self.smartdoc = SmartDocDataset(**kwargs)
            target_dataset.append('smartdoc')
        if use_private:
            self.private = PrivateDataset(**kwargs)
            target_dataset.append('private')

        self.pool = D.get_files(
            DIR.parent / 'data' / 'docpool', suffix=['.jpg'])
        target_dataset.append('pool')
        self.target_dataset = target_dataset

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'indoor_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            dataset.append((img_path, None))
        return dataset

    def _random_get_doc_image(self):
        tgt = np.random.choice(self.target_dataset)
        if tgt == 'cord':
            return D.imwarp_quadrangle(*self.cord[np.random.randint(len(self.cord))])
        elif tgt == 'midv500':
            return D.imwarp_quadrangle(*self.midv500[np.random.randint(len(self.midv500))])
        elif tgt == 'midv2019':
            return D.imwarp_quadrangle(*self.midv2019[np.random.randint(len(self.midv2019))])
        elif tgt == 'midv2020':
            return D.imwarp_quadrangle(*self.midv2020[np.random.randint(len(self.midv2020))])
        elif tgt == 'smartdoc':
            return D.imwarp_quadrangle(*self.smartdoc[np.random.randint(len(self.smartdoc))])
        elif tgt == 'private':
            return D.imwarp_quadrangle(*self.private[np.random.randint(len(self.private))])
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

        if np.random.rand() < 0.5:
            # Generate doc image on indoor dataset.
            idx = np.random.randint(len(self))
            img_path, _ = self.dataset[idx]
            img = D.imread(img_path)
            if img is None:
                return self.__getitem__(idx)
            poly = self._generate_random_quadrant_points()
        else:
            # Generate doc image on midv dataset.
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


class DocAlignerDataset:

    def __init__(
        self,
        root: Union[str, Path] = None,
        image_size: Tuple[int, int] = (256, 256),
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        aug_func: Callable = None,
        aug_ratio: float = 0.0,
        length_of_dataset: int = 100000,
        fuse_dataset: List[str] = [
            'midv500', 'midv2019', 'midv2020', 'cord', 'sync', 'smartdoc'],
        fuse_ratio: List[float] = [0.2, 0.2, 0.1, 0.1, 0.3, 0.1],
        edge_width: int = 3,
        output_tensor: bool = True,
    ) -> None:
        self.fuse_ratio = fuse_ratio
        self.edge_width = edge_width
        self.output_tensor = output_tensor
        self.length_of_dataset = length_of_dataset

        # Dataset settings
        ds_settings = {
            'root': root,
            'image_size': image_size,
            'interpolation': interpolation,
            'aug_func': aug_func,
            'aug_ratio': aug_ratio,
        }

        dataset = []
        for d in fuse_dataset:
            if d not in ['midv500', 'midv2019', 'midv2020', 'cord', 'sync', 'smartdoc']:
                raise ValueError(f'Unknown dataset: {d}')
            if d == 'midv500':
                dataset.append(MIDV500Dataset(**ds_settings))
            if d == 'midv2019':
                dataset.append(MIDV2019Dataset(**ds_settings))
            if d == 'midv2020':
                dataset.append(MIDV2020Dataset(**ds_settings))
            if d == 'cord':
                dataset.append(CordDataset(**ds_settings))
            if d == 'sync':
                dataset.append(SyncDataset(**ds_settings))
            if d == 'smartdoc':
                dataset.append(SmartDocDataset(mode='train', **ds_settings))
        self.dataset = dataset

    def __len__(self) -> int:
        return self.length_of_dataset

    def to_tensor(self, img, box, poly, edge, edge_mask):
        poly = D.Polygon(poly).normalize(
            w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
        box = D.Box(box).normalize(
            w=img.shape[1], h=img.shape[0]).numpy().astype('float32')
        img = np.transpose(img.astype('float32'), (2, 0, 1)) / 255.0
        edge = edge.astype('float32') / 255.0
        edge_mask = edge_mask.astype('float32')
        return img, box, poly, edge, edge_mask

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        idx = idx % len(self)
        d_idx = np.random.choice(len(self.dataset), p=self.fuse_ratio)
        f_idx = np.random.randint(len(self.dataset[d_idx]))
        img, poly = self.dataset[d_idx][f_idx]
        edge = cv2.fillPoly(
            np.zeros_like(img),
            [poly.astype('int32')],
            color=(255, 255, 255)
        )
        edge = D.imresize(edge, (edge.shape[0] // 2, edge.shape[1] // 2))
        edge = D.imgrandient(D.imbinarize(edge), ksize=self.edge_width)
        edge_mask = D.imdilate(edge, ksize=self.edge_width) > 0

        # Polygon -> Bounding Box
        box = D.Polygon(poly).to_box(box_mode='XYWH').numpy()

        if self.output_tensor:
            return self.to_tensor(img, box, poly, edge, edge_mask)

        return img, box, poly, edge, edge_mask
