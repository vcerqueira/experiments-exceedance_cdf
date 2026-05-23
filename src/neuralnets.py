from typing import Optional

from neuralforecast.models import (KAN,
                                   TFT,
                                   NBEATS,
                                   TiDE,
                                   NLinear,
                                   MLP,
                                   DLinear,
                                   NHITS,
                                   PatchTST,
                                   DeepNPTS)


class BaseModelsConfig:

    @classmethod
    def get_nf_models(cls,
                      horizon: int,
                      input_size: int,
                      try_mps: bool = True,
                      limit_epochs: bool = False,
                      limit_val_batches: Optional[int] = None):

        engine = 'mps' if try_mps else 'cpu'

        config = {
            'input_size': input_size,
            'h': horizon,
            'enable_checkpointing': True,
            'accelerator': engine}

        if limit_epochs:
            config['max_steps'] = 2

        if limit_val_batches is not None:
            config['limit_val_batches'] = limit_val_batches

        models = [
            NBEATS(**config),
            NHITS(**config),
            MLP(**config),
            # MLP(**config, num_layers=3),
            TiDE(**config),
            KAN(**config),
            # TimeMixer(**config, n_series = 1),
            TFT(**config, scaler_type='standard'),
            NLinear(**config),
            DLinear(**config),
            PatchTST(**config),
            DeepNPTS(**config),
        ]

        return models
