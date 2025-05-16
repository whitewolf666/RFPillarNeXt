import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
from thop import profile, clever_format

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--test_interval', type=int, default=100,
                        help='specify the test interval for inference')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--fuse', type=bool, default=False, help='fuse the Conv2d and BN')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    if args.fuse:
        model.dense_head.fuse()
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            data_dict_copy = data_dict.copy()
            flops, params = profile(model, inputs=(data_dict_copy, ), verbose=False)
            flops, params = clever_format([flops, params], "%.6f")
            pred_dicts, _ = model.forward(data_dict)

    test_interval = args.test_interval
    elapsed_time = 0
    for _ in range(test_interval):
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                logger.info(f'Visualized sample index: \t{idx + 1}')
                torch.cuda.synchronize()
                t1 = time.time()
                pred_dicts, _ = model.forward(data_dict)
                torch.cuda.synchronize()
                t2 = time.time()
                elapsed_time += (t2 - t1) / test_interval

    logger.info(str(elapsed_time) + ' seconds, ' + str(1 / elapsed_time) + ' FPS')
    logger.info(f'flops: {flops}, params: {params}')

    logger.info('inference done.')


if __name__ == '__main__':
    main()
