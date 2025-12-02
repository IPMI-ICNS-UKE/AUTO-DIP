#!/usr/bin/env python

import os
from pathlib import Path

import polars as pl
import yaml
from load_fmd import FMDDataset


def create_configs_for_one_image(
    params_image: dict, exp_name: str, img_name: str, save_folder: Path
):
    """create config files for one image and all parameter combinations

    Args:
        params_image (dict): parameters for the image
        exp_name (str): MLflow experiment name
        img_name (str): unique name for the image
        save_folder (Path): path to save the resulting configs
    """
    save_path_image = save_folder / img_name
    if not save_path_image.exists():
        os.makedirs(save_path_image)

    for depth in [4, 5, 6, 7, 8]:
        for width in [16, 32, 64, 128, 256, 512, "asc_to512"]:
            for skip in [0, 4]:
                save_path_config = save_path_image / f"s{skip}_w{width}_d{depth}.yaml"
                create_config(
                    params_image,
                    exp_name,
                    img_name,
                    skip,
                    width,
                    depth,
                    save_path_config,
                )


def create_config(
    params_image: dict,
    exp_name: str,
    img_name: str,
    skip: int,
    width: int,
    depth: int,
    save_path: Path,
):
    """create one config for one image and one parameter configuration

    Args:
        params_image (dict): paramters regarding the image
        exp_name (str): MLflow experiment name
        img_name (str): unique name for the image
        skip (int): number of skip connections
        width (int): width of the network
        depth (int): depth of the network
        save_path (Path): path to save the config file

    """
    params_save = {
        "verbosity": 0,
        "mlflow": True,
        "mlflow_log_loss_types": ["mse", "ssim", "psnr", "l1", "lpips"],
        "mlflow_exp_name": exp_name,
        "mlflow_run_name": f"{img_name}_s{skip}_w{width}_d{depth}",
        "mlflow_log_images": False,
        "save_imgs": True,
        "save_path": str(Path("./search_results/") / exp_name),
    }

    if width == "asc_to512":
        width = [2**i for i in range(10 - depth, 10)]
        skip = [skip for i in range(10 - depth, 10)]

    params_net = {
        "num_iter": 3000,
        "skip_n11": skip,
        "num_scales": depth,
        "skip_n33d": width,
        "skip_n33u": width,
    }

    params = {"image": params_image, "net": params_net, "save_and_log": params_save}

    def represent_list(dumper, data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", list(data), flow_style=True
        )

    # Custom Dumper class to disable aliases
    class NoAliasDumper(yaml.Dumper):
        def ignore_aliases(self, data):
            return True  # This disables the automatic use of aliases (anchors)

    # Add the custom representation for lists
    yaml.add_representer(list, represent_list)

    with save_path.open("w") as f:
        yaml.dump(params, f, Dumper=NoAliasDumper)


if __name__ == "__main__":
    datasets = [
        FMDDataset(
            root_dir=Path("./data/fmd/"),
            select_imgs_file=Path("./data/data_selection/fmd_calibration.csv"),
        ),
        FMDDataset(
            root_dir=Path("./data/fmd/"),
            select_imgs_file=Path("./data/data_selection/fmd_validation.csv"),
        ),
    ]

    for dataset in datasets:
        config_path = Path("./configs/") / dataset.name

        exp_name = dataset.name

        sel_imgs_df = dataset.df
        sel_imgs_df = sel_imgs_df.filter(pl.col("image_quality") == "noisy")

        for img_row in sel_imgs_df.iter_rows(named=True):
            img_name = dataset.create_img_name(img_row)

            gt_row = dataset.get_params_for_dip(img_row)

            noisy_path = img_row["path"]
            gt_path = gt_row["path_gt"]

            params_image = {}
            params_image["path"] = noisy_path
            params_image["path_gt"] = gt_path

            if "crop_region" in gt_row:
                params_image["crop_region"] = gt_row["crop_region"]
            if "frame_range" in gt_row:
                params_image["frame_range"] = gt_row["frame_range"]

            create_configs_for_one_image(
                params_image=params_image,
                exp_name=exp_name,
                img_name=img_name,
                save_folder=config_path,
            )
