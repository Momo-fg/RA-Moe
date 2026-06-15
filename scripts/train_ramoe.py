import argparse
from distutils.util import strtobool
from pathlib import Path
import sys

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ramoe.training.ramoe_trainer import main_SKD_TSTSAN_with_Aug_with_SKD


def load_config(config_path):
    if config_path is None:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required when using --config. Install requirements.txt first.")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config_defaults(args, cfg):
    dataset_cfg = cfg.get("dataset", {})
    assets_cfg = cfg.get("assets", {})
    train_cfg = cfg.get("train", {})
    output_cfg = cfg.get("output", {})

    if not args.main_path:
        args.main_path = dataset_cfg.get("loso_root", "")
    if not args.pre_trained_model_path:
        args.pre_trained_model_path = assets_cfg.get("skd_tstsan", "")
    if not args.magnet_model_path:
        args.magnet_model_path = assets_cfg.get("magnet", "")
    if not args.exp_name:
        args.exp_name = output_cfg.get("experiment_name", "casme2_ramoe")
    if not args.output_dir:
        args.output_dir = output_cfg.get("output_dir", str(ROOT / "outputs" / args.exp_name))
    if not args.regionnet_dir:
        args.regionnet_dir = output_cfg.get("regionnet_dir", str(ROOT / "checkpoints" / "regionnet"))

    args.dataset = dataset_cfg.get("name", args.dataset)
    args.class_num = int(dataset_cfg.get("num_classes", args.class_num))
    args.learning_rate = float(train_cfg.get("learning_rate", args.learning_rate))
    args.batch_size = int(train_cfg.get("batch_size", args.batch_size))
    args.seed = int(train_cfg.get("seed", args.seed))
    args.max_iter = int(train_cfg.get("max_iter", args.max_iter))
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help="optional YAML config path")
    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--pre_trained', type=strtobool, default=True)
    parser.add_argument('--Aug_COCO_pre_trained', type=strtobool, default=True)
    parser.add_argument('--save_model', type=strtobool, default=True)
    parser.add_argument('--pre_trained_model_path', type=str, default="", help="path to the model weights pre-trained on macro-expression dataset")
    parser.add_argument('--main_path', type=str, default="", help="path to the dataset directory")
    parser.add_argument('--output_dir', type=str, default="", help="directory for logs and trained RA-MoE checkpoints")
    parser.add_argument('--regionnet_dir', type=str, default="", help="directory containing RegionNet checkpoints")
    parser.add_argument('--magnet_model_path', type=str, default="", help="path to the motion magnification checkpoint")
    parser.add_argument('--dataset', type=str, default="casme2", choices=["casme2"])
    parser.add_argument('--exp_name', type=str, default="", help="name of the folder to save experimental results")
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--model', type=str, default="SKD_TSTSAN")
    parser.add_argument('--loss_function', type=str, default="FocalLoss_weighted")
    parser.add_argument('--class_num', type=int, default=5)

    parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float, help='weight of feature loss')

    parser.add_argument('--Aug_alpha', type=float, default=2)

    config = parser.parse_args()
    config = apply_config_defaults(config, load_config(config.config))

    main_SKD_TSTSAN_with_Aug_with_SKD(config)
