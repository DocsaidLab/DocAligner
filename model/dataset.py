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
                scale_limit=[-0.4, 0.2],
                border_mode=cv2.BORDER_CONSTANT),
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


class CordDataset(BaseDataset):

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'cord_v0_train_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            gt = data['quad']
            dataset.append((img_path, gt))
        return dataset


class SyncDataset(BaseDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        kwargs.update({'aug_ratio': 0.0})
        self.midv = MIDV500Dataset(**kwargs)
        self.cord = CordDataset(**kwargs)
        self.pool = D.get_files(
            DIR.parent / 'data' / 'docpool', suffix=['.jpg'])

    def _build(self):
        ds = D.load_json(DIR.parent / 'data' / 'indoor_dataset.json')
        dataset = []
        for data in D.Tqdm(ds):
            img_path = self.root / data['img_path']
            dataset.append((img_path, None))
        return dataset

    def _random_get_doc_image(self):
        tgt = np.random.choice(['midv', 'cord', 'pool'], p=[0.1, 0.1, 0.8])
        if tgt == 'cord':
            return D.imwarp_quadrangle(*self.cord[np.random.randint(len(self.cord))])
        elif tgt == 'midv':
            return D.imwarp_quadrangle(*self.midv[np.random.randint(len(self.midv))])
        else:
            img = self.pool[np.random.randint(len(self.pool))]
            return D.imread(img)

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
            idx = np.random.randint(len(self))
            img_path, _ = self.dataset[idx]
            img = D.imread(img_path)
            if img is None:
                return self.__getitem__(idx)
            poly = self._generate_random_quadrant_points()
        else:
            img, poly = self.midv[np.random.randint(len(self.midv))]
            poly = D.Polygon(poly).normalize(
                img.shape[1], img.shape[0]).numpy()

        doc_img = self._random_get_doc_image()
        sync_img, poly = self._paste_doc_image(img, poly, doc_img)

        if self.image_size is not None:
            sync_img, poly = self._resize_poly(sync_img, poly)

        sync_img, poly = self.aug_func(image=sync_img, keypoints=poly)
        return sync_img, poly


class DocAlignedDataset:

    def __init__(
        self,
        root: Union[str, Path] = None,
        image_size: Tuple[int, int] = (256, 256),
        interpolation: Union[str, int, INTER] = INTER.BILINEAR,
        aug_func: Callable = None,
        aug_ratio: float = 0.0,
        length_of_dataset: int = 100000,
        fuse_dataset: List[str] = ['midv500', 'cord', 'sync'],
        fuse_ratio: List[float] = [0.4, 0.2, 0.4],
        edge_width: int = 3,
        output_tensor: bool = True,
    ) -> None:
        self.fuse_ratio = fuse_ratio
        self.edge_width = edge_width
        self.output_tensor = output_tensor
        self.length_of_dataset = length_of_dataset
        ds_settings = {
            'root': root,
            'image_size': image_size,
            'interpolation': interpolation,
            'aug_func': aug_func,
            'aug_ratio': aug_ratio,
        }

        dataset = []
        for d in fuse_dataset:
            if d not in ['midv500', 'midv2019', 'cord', 'sync']:
                raise ValueError(f'Unknown dataset: {d}')
            if d == 'midv500':
                dataset.append(MIDV500Dataset(**ds_settings))
            if d == 'midv2019':
                dataset.append(MIDV2019Dataset(**ds_settings))
            if d == 'cord':
                dataset.append(CordDataset(**ds_settings))
            if d == 'sync':
                dataset.append(SyncDataset(**ds_settings))
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
