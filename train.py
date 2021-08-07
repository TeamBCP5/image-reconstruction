import argparse
import os
from importlib import import_module
from utils import Flags, print_arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_type",
        type=str,
        default="all",
        help="""
        학습 방식 설정
         - 'all': 
            Main 모델(Pix2Pix) 학습 후 Postprocessing 모델(HINet) 학습
            Postprocessing 모델(HINet)의 경우 다음의 두 단계로 구성
             - Phase 1. Input: 주어진 데이터 input, Label: 주어진 데이터 label
             - Phase 2. Input: Main 모델(Pix2Pix) output, Label: 주어진 데이터 label
         - 'pix2pix': Pix2Pix 모델 개별 학습
         - 'hinet': HINet 모델 개별 학습
        """
    )
    parser.add_argument(
        "--config_pix2pix",
        type=str,
        default="./configs/Pix2Pix.yaml",
        help="Pix2Pix 모델 configuration 파일 경로",
    )
    parser.add_argument(
        "--config_hinet_phase1",
        type=str,
        default="./configs/HINet_phase1.yaml",
        help="HINet 모델(phase1) configuration 파일 경로",
    )
    parser.add_argument(
        "--config_hinet_phase2",
        type=str,
        default="./configs/HINet_phase2.yaml",
        help="HINet 모델(phase2) configuration 파일 경로",
    )
    args = parser.parse_args()
    
    # exception
    if args.train_type not in ["pix2pix", "hinet", "all"]:
        raise ValueError("Choose 'train_type' one of 'all', 'pix2pix', 'hinet'")

    if not os.path.isfile(args.config_pix2pix):
        raise ValueError(f"There's no file '{args.config_pix2pix}'")
    
    if not os.path.isfile(args.config_hinet_phase1):
        raise ValueError(f"There's no file '{args.config_hinet_phase1}'")

    if not os.path.isfile(args.config_hinet_phase2):
        raise ValueError(f"There's no file '{args.config_hinet_phase2}'")

    # train        
    if args.train_type == 'all': # train all necessary models
        pix2pix_args = Flags(args.config_pix2pix).get()
        hinet_phase1_args = Flags(args.config_hinet_phase1).get()
        hinet_phase2_args = Flags(args.config_hinet_phase2).get()

        if pix2pix_args.network.name != 'pix2pix':
            raise ValueError("Network work name is not equal to 'pix2pix'. check configuration file.")
        if hinet_phase1_args.network.name != 'hinet':
            raise ValueError("Network work name is not equal to 'hinet'. check configuration file.")
        if hinet_phase2_args.network.name != 'hinet':
            raise ValueError("Network work name is not equal to 'hinet'. check configuration file.")

        # train Pix2Pix
        train_module = getattr(
            import_module(f"train_modules.train_{pix2pix_args.network.name}"), "train"
            )
        print("<<< Train Pix2Pix >>>")
        print_arguments(pix2pix_args)
        train_module(pix2pix_args)

        # train HINet
        train_module = getattr(
            import_module(f"train_modules.train_{hinet_phase1_args.network.name}"), "train"
            )
        # phase 1
            # input: origin input image
            # label: origin label image
        print("<<< Train HINet - Phase 1 >>>")
        print_arguments(hinet_phase1_args)
        train_module(hinet_phase1_args, phase=1)
        # phase 2
            # input: main model(pix2pix) output image
            # label: origin label image
        print("<<< Train HINet - Phase 2 >>>")
        print_arguments(hinet_phase2_args)
        train_module(hinet_phase2_args, phase=2) 
    
    # train pix2pix in single
    elif args.train_type == 'pix2pix':
        args = Flags(args.config_pix2pix).get()
        if args.network.name != 'pix2pix':
            raise ValueError("Network work name is not equal to 'pix2pix'. Check configuration file.")
        train_module = getattr(
            import_module(f"train_modules.train_{args.network.name}"), "train"
            )
        print("<<< Train Pix2Pix in Single >>>")
        print_arguments(args)
        train_module(args)
    
    # train hinet in single
    elif args.train_type == 'hinet':
        args = Flags(args.hinet_phase1_args).get()
        if args.network.name != 'hinet':
            raise ValueError("Network work name is not equal to 'hinet'. heck configuration file.")
        train_module = getattr(
            import_module(f"train_modules.train_{args.network.name}"), "train"
            )
        print("<<< Train HINet in Single >>>")
        print_arguments(args)
        train_module(args, phase=1)
