from pprint import pprint

import docsaidkit as D
import docsaidkit.torch as DT

from . import dataset as ds
from . import network as net

DIR = D.get_curdir(__file__)


def main_docaligner_train(cfg_name: str):
    model, cfg = DT.load_model_from_config(
        root=DIR, stem='config', cfg_name=cfg_name, network=net)
    trainer = DT.build_trainer(cfg, root=DIR)
    train_data, valid_data = DT.build_dataset(cfg, ds)

    # -- Log model meta data -- #
    macs, params = DT.get_model_complexity_info(
        model,
        input_res=(3, *cfg.common.image_size),
        as_strings=False,
        print_per_layer_stat=False
    )
    meta_data = DT.get_meta_info(macs, params)
    D.dump_json(
        meta_data,
        DIR / cfg.name / cfg.name_ind / 'logger' / 'model_meta_data.json'
    )
    pprint(meta_data)
    # ------------------------- #

    trainer.fit(
        model,
        train_dataloaders=train_data,
        val_dataloaders=valid_data
    )
