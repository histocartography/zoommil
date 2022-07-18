
class Config:
    """Contains all configuration parameters."""
    def __init__(self, config_dict):

        # general config
        self.seed: int = config_dict.get('seed', 1)
        self.max_epochs: int = config_dict.get('max_epochs', 100)
        self.save_path: str = config_dict.get('save_path', None)
        
        # optimizer config
        self.lr: float = config_dict.get('lr', 1e-4)
        self.reg: float = config_dict.get('reg', 0.)
        self.scheduler_decay_rate: float = config_dict.get('scheduler_decay_rate', 0.5)
        self.scheduler_patience: int = config_dict.get('scheduler_patience', 5)
        
        # dataset and dataloader config
        self.data_path: str = config_dict.get('data_path', None)
        self.csv_path: str = config_dict.get('csv_path', None)
        self.split_path: str = config_dict.get('split_path', None)
        self.label_col: str = config_dict.get('label_col', "type")
        self.low_mag: str = config_dict.get('low_mag', None)
        self.mid_mag: str = config_dict.get('mid_mag', None)
        self.high_mag: str = config_dict.get('high_mag', None)
        self.is_weighted_sampler: bool = config_dict.get('is_weighted_sampler', True)
        self.num_workers: int = config_dict.get('num_workers', 8)
        
        # model
        self.n_cls: int = config_dict.get('n_cls', 3)
        self.k_sample: int = config_dict.get('k_sample', 12)
        self.k_sigma: float = config_dict.get('k_sigma', 0.002)
        self.drop_out: float = config_dict.get('drop_out', None)
        self.in_feat_dim: int = config_dict.get('in_feat_dim', 1024)
        
        # model saver config
        self.save_metric: str = config_dict.get('save_metric', "loss")
