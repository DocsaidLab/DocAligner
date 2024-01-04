from pprint import pprint

import cv2
import docsaidkit as D
import docsaidkit.torch as DT
import torch

from . import dataset as ds
from . import network as net

DIR = D.get_curdir(__file__)

cv2.setNumThreads(0)

torch.set_num_threads(1)


def main_docaligner_train(cfg_name: str):
    model, cfg = DT.load_model_from_config(
        root=DIR, stem='config', cfg_name=cfg_name, network=net)
    train_data, valid_data = DT.build_dataset(cfg, ds)

    if cfg.lr_scheduler.name == 'PolynomialLRWarmup':
        total_iters = cfg.trainer.max_epochs * \
            len(train_data.dataset) // cfg.common.batch_size
        warmup_iters = int(total_iters * 0.1)
        cfg.lr_scheduler.options.update({
            'warmup_iters': warmup_iters,
            'total_iters': total_iters,
        })

    trainer = DT.build_trainer(cfg, root=DIR)

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

    restore_all_states = getattr(cfg.common, 'restore_all_states', False)
    trainer.fit(
        model,
        train_dataloaders=train_data,
        val_dataloaders=valid_data,
        ckpt_path=cfg.common.checkpoint_path if restore_all_states else None,
    )
