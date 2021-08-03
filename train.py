import argparse
from importlib import import_module
from utils import Flags

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        default="./configs/Pix2Pix.yaml",
        help="train config file path",
    )
    args = parser.parse_args()
    args = Flags(args.config_file).get()

    if args.network.name not in ["pix2pix", "hinet"]:
        raise ValueError("Choose network one of 'pix2pix' or 'hinet'")

    print("=" * 100)
    print(args)
    print("=" * 100)
    print()

    train_module = getattr(
        import_module(f"train_modules.train_{args.network.name}"), "train"
    )
    train_module(args)
