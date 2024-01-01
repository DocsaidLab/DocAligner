import docsaidkit as D
import numpy as np
import pandas as pd
from tabulate import tabulate

from ..docaligner import DocAligner, ModelType

DIR = D.get_curdir(__file__)

DATA_DIR = DIR / 'passport_benchmark_dataset'


def main(model_type: ModelType, model_cfg: str):

    model = DocAligner(model_type=model_type, model_cfg=model_cfg)
    scores = []

    gts = D.PowerDict.load_json(DATA_DIR / 'gt.json')
    for _, gt in D.Tqdm(gts.items()):
        img = D.imread(DATA_DIR / gt.path)
        poly = np.array(gt.polygon).reshape(4, 2)
        pred_poly = model(img)
        pred_poly = pred_poly.doc_polygon
        if pred_poly is not None and len(pred_poly) == 4:
            poly = D.order_points_clockwise(poly)
            pred_poly = D.order_points_clockwise(pred_poly)
            mask_iou = D.polygon_iou(D.Polygon(pred_poly), D.Polygon(poly))
        else:
            mask_iou = 0
        scores.append(mask_iou)

    # Prepare data for tabulation
    table_data = [[sum(scores) / len(scores)]]
    headers = ["Mask IoU"]
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
