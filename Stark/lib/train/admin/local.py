class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/usr/mvl2/esdft/Stark'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/usr/mvl2/esdft/Stark/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/usr/mvl2/esdft/Stark/pretrained_networks'
        self.lasot_dir = '/usr/mvl2/esdft/Stark/data/lasot'
        self.got10k_dir = '/usr/mvl2/esdft/Stark/data/got10k'
        self.lasot_lmdb_dir = '/usr/mvl2/esdft/Stark/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/usr/mvl2/esdft/Stark/data/got10k_lmdb'
        self.trackingnet_dir = '/usr/mvl2/esdft/Stark/data/trackingnet'
        self.trackingnet_lmdb_dir = '/usr/mvl2/esdft/Stark/data/trackingnet_lmdb'
        self.coco_dir = '/usr/mvl2/esdft/Stark/data/coco'
        self.coco_lmdb_dir = '/usr/mvl2/esdft/Stark/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/usr/mvl2/esdft/Stark/data/vid'
        self.imagenet_lmdb_dir = '/usr/mvl2/esdft/Stark/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
