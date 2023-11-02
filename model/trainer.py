import docsaidkit as D
import docsaidkit.torch as DT

from . import dataset as ds
from . import network as net

DIR = D.get_curdir(__file__)


def main_docalign_train(cfg_name: str):
    model, cfg = DT.load_model_from_config(
        root=DIR, stem='config', cfg_name=cfg_name, network=net)
    trainer = DT.build_trainer(cfg, root=DIR)
    train_data, valid_data = DT.build_dataset(cfg, ds)
    trainer.fit(
        model,
        train_dataloaders=train_data,
        val_dataloaders=valid_data
    )
