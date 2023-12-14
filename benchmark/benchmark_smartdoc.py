import docsaidkit as D
import pandas as pd
from tabulate import tabulate

from ..api import DocAligner, ModelType
from ..model.dataset import SmartDocDataset


def main():

    model = DocAligner(model_type=ModelType.HeatmapBased)
    dataset = SmartDocDataset(root='/data/Dataset', mode='val')

    doc_types, mask_ious = [], []
    for i in D.Tqdm(range(len(dataset))):
        img, poly, key = dataset[i]
        pred_poly = model(img).polygon
        if len(pred_poly) == 4:
            poly = D.order_points_clockwise(poly)
            pred_poly = D.order_points_clockwise(pred_poly)
            mask_iou = D.jaccard_index(
                pred_poly, poly, image_size=(2970, 2100))
        else:
            mask_iou = 0

        doc_types.append(key)
        mask_ious.append(mask_iou)

    df = pd.DataFrame({
        'DocType': doc_types,
        'IoU': mask_ious,
    })

    grp_question = df.groupby(by='DocType', group_keys=False)

    n_ds = grp_question['DocType'].count() \
        .reset_index(name='Number') \
        .set_index('DocType')

    iou = grp_question['IoU'].mean() \
        .reset_index(name='IoU') \
        .set_index('DocType')

    overall = pd.DataFrame({
        'Number': [len(mask_ious)],
        'IoU': [sum(mask_ious)/len(mask_ious)],
    }, index=['Overall'])

    df = pd.concat([n_ds, iou], axis=1)
    df = pd.concat([df, overall], axis=0)

    table = tabulate(
        df.T,
        headers='keys',
        tablefmt='psql',
        numalign='right',
        stralign='right',
        floatfmt='.4f',
        intfmt='d'
    )

    print('\n')
    print(table)
    print('\n')
