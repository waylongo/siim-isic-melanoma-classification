# SETTINGS
import argparse

parser = argparse.ArgumentParser(description="Melanoma Parameters")
parser.add_argument('--image_size', type=int, help='Image Size')
parser.add_argument('--efficient_type', type=int, help='EfficientNet Type')
parser.add_argument('--batch_size', type=int, help='Batch Size')
parser.add_argument('--random_seed', type=int, help='Random Seed')

parser.add_argument('--epoch', type=int, default=25, help='Epoch')
parser.add_argument('--label_smoothing', type=float, default=0.02, help='Label Smoothing')
parser.add_argument('--num_workers', type=int, default=16, help='Num Workers')
parser.add_argument('--es_patience', type=int, default=7, help='Early Stopping Patience')
parser.add_argument('--tta', type=int, default=11, help='Test Time Augmentation')
parser.add_argument('--use_2018', type=bool, default=True, help='Use 2018 data')
parser.add_argument('--use_2019', type=bool, default=False, help='Use 2019 data')
args = parser.parse_args()

IMG_SIZE = args.image_size
EFFICIENT_TYPE = args.efficient_type
BATCH_SIZE = args.batch_size
RANDOM_SEED = args.random_seed
EPOCHS = args.epoch
LABEL_SMOOTHING = args.label_smoothing
NUM_WORKERS = args.num_workers
ES_PATIENCE = args.es_patience
TTA = args.tta
USE_2018 = args.use_2018
USE_2019 = args.use_2019