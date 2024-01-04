import docsaidkit as D
import pandas as pd
from tabulate import tabulate

from ..docaligner import DocAligner, ModelType
from ..model.dataset import SmartDocDataset

DIR = D.get_curdir(__file__)


def main(model_type: ModelType, model_cfg: str):

    model_type = ModelType.obj_to_enum(model_type)
    model = DocAligner(model_type=model_type, model_cfg=model_cfg)
    dataset = SmartDocDataset(root='/data/Dataset', mode='val', train_ratio=0)

    doc_types, mask_ious = [], []
    for i in D.Tqdm(range(len(dataset))):
        img, poly, key = dataset[i]
        pred_poly = model(img).doc_polygon
        if pred_poly is not None and len(pred_poly) == 4:
            poly = D.order_points_clockwise(poly)
            pred_poly = D.order_points_clockwise(pred_poly)
            mask_iou = D.jaccard_index(
                pred_poly, poly, image_size=(2970, 2100))

            if not (fp := DIR / 'test_output').is_dir():
                fp.mkdir(parents=True)

            if mask_iou < 0.9:
                export = D.draw_polygon(img, pred_poly, color=(0, 255, 0))
                export = D.draw_polygon(export, poly, color=(255, 0, 0))
                D.imwrite(export, fp / f'{mask_iou:.4f}.jpg')

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
