from pathlib import Path
from typing import Tuple, Union

import docsaidkit as dsk
import lightning.pytorch.callbacks as pl_callbacks
import lightning.pytorch.loggers as pl_loggers
import natsort
import network as net
import torch.nn as nn
from fire import Fire
from lightning import Trainer
from torch.utils.data import DataLoader

import dataset as ds

DIR = dsk.get_curdir(__file__)

T = dsk.now(fmt='%Y-%m-%d %H:%M:%S')


def load_model_from_config(cfg_name: Union[str, Path]) -> Tuple[nn.Module, dsk.PowerDict]:

    cfg = dsk.PowerDict.load_yaml(DIR / 'config' / f'{cfg_name}.yaml')
    ind = cfg.common.restore_ind if cfg.common.restore_ind != "" else T
    cfg.update({'name': str(cfg_name), 'name_ind': ind})

    # check key
    if 'common' not in cfg:
        raise KeyError('Key "common" is not in config file.')

    # check model
    if 'model' not in cfg:
        raise KeyError('Key "model" is not in config file.')
    net_name = cfg.model.name

    # load model
    if cfg.common.is_restore:
        _ckpt = cfg.common.restore_ckpt
        _path = Path().joinpath(cfg_name, ind, 'checkpoint', 'model')
        if _ckpt is None or _ckpt == '':
            _candi_model = [i for i in dsk.get_files(_path, suffix=['.ckpt']) if 'last' in i.stem]
            _ckpt = natsort.os_sorted(_candi_model)[-1]
        checkpoint_path = str(DIR / _path / _ckpt)
        model = getattr(net, net_name).load_from_checkpoint(checkpoint_path, cfg=cfg, strict=False)
        print(f'MODEL Load from checkpoint {dsk.colorstr(checkpoint_path)}... Done.', flush=True)
    else:
        model = getattr(net, net_name)(cfg=cfg)

    return model, cfg


def build_callback(cfg: dsk.PowerDict):
    callbacks = []
    for callback in cfg.callbacks:
        if callback.name == 'ModelCheckpoint':
            dirpath = Path().joinpath(cfg.name, cfg.name_ind, 'checkpoint', 'model')
            callback.options.update({'dirpath': str(dirpath)})
        options = getattr(callback, 'options', {})
        callbacks.append(getattr(pl_callbacks, callback.name)(**options))
    return callbacks


def build_logger(cfg: dsk.PowerDict):
    cfg.logger.options.update({
        'save_dir': str(Path().joinpath(cfg.name, cfg.name_ind, cfg.logger.options.save_dir))
    })
    logger = getattr(pl_loggers, cfg.logger.name)(**cfg.logger.options)
    if not (log_dir := Path(cfg.logger.options.save_dir)).is_dir():
        log_dir.mkdir(parents=True)
    cfg.to_yaml(Path(cfg.logger.options.save_dir) / 'config.yaml')
    return logger


def build_dataset(cfg: dsk.PowerDict):
    ds_loader_train_opts = cfg.dataloader.train_options
    ds_loader_valid_opts = cfg.dataloader.valid_options
    ds_loader_train_opts.update({'batch_size': cfg.common.batch_size})
    ds_loader_valid_opts.update({'batch_size': cfg.common.batch_size})
    ds_train_name, ds_train_opts = cfg.dataset.train_options.values()
    ds_valid_name, ds_valid_opts = cfg.dataset.valid_options.values()
    ds_train_opts.update({'image_size': cfg.common.image_size})
    ds_valid_opts.update({'image_size': cfg.common.image_size})
    ds_train = getattr(ds, ds_train_name)(**ds_train_opts)
    ds_valid = getattr(ds, ds_valid_name)(**ds_valid_opts)
    train_data = DataLoader(dataset=ds_train, **ds_loader_train_opts)
    valid_data = DataLoader(dataset=ds_valid, **ds_loader_valid_opts)
    return train_data, valid_data


def main(cfg_name: str):
    model, cfg = load_model_from_config(cfg_name)
    train_data, valid_data = build_dataset(cfg)
    callbacks = build_callback(cfg)
    logger = build_logger(cfg)
    trainer = Trainer(**cfg.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_data, val_dataloaders=valid_data)


if __name__ == '__main__':
    Fire(main)
