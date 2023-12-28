import docsaidkit as D
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..docaligner import DocAligner, ModelType

DIR = D.get_curdir(__file__)

DATA_DIR = DIR / 'idcard_benchmark_dataset'

TARG_IDCard_DATA_FOLDER = [
    'env_dark_background_bright',
    'env_dark_background_complex',
    'env_dark_background_dark',
    'env_normal_background_bright',
    'env_normal_background_complex',
    'env_normal_background_computer',
    'env_normal_background_dark',
]


def main(model_type: ModelType, model_cfg: str):

    model_type = ModelType.obj_to_enum(model_type)
    model = DocAligner(model_type=model_type, model_cfg=model_cfg)
    env_scores = []
    for env_name in TARG_IDCard_DATA_FOLDER:
        gts = D.PowerDict.load_json(DATA_DIR / env_name / f'{env_name}.json')
        single_env_scores = []
        for _, gt in D.Tqdm(gts.items()):
            img = D.imread(DATA_DIR / env_name / gt.imagePath)
            poly = np.array(gt.polygons)
            result = model(img)
            pred_poly = result.doc_polygon
            if pred_poly is not None and len(pred_poly) == 4:
                poly = D.order_points_clockwise(poly)
                pred_poly = D.order_points_clockwise(pred_poly)
                mask_iou = D.polygon_iou(D.Polygon(pred_poly), D.Polygon(poly))
            else:
                mask_iou = 0
            single_env_scores.append(mask_iou)
        env_scores.append(sum(single_env_scores) / len(single_env_scores))

    # Prepare data for tabulation
    table_data = []
    for env_name, score in zip(TARG_IDCard_DATA_FOLDER, env_scores):
        table_data.append([env_name, score])

    headers = ["Environment", "Mask IoU"]
    table = tabulate(
        table_data,
        headers=headers,
        tablefmt="psql",
        numalign='right',
        stralign='right',
        floatfmt='.4f',
        intfmt='d'
    )

    print('\n')
    print(table)
    print('\n')
