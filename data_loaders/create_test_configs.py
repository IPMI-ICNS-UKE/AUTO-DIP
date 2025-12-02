#!/usr/bin/env python

import os
from pathlib import Path

import polars as pl
import yaml
from load_biosr import BioSRDataset
from load_fmd import FMDDataset
from load_fmd_test_mix import FMDTestMixDataset
from load_hagen_test import HagenTestDataset
from load_shah import ShahDataset
from load_w2s import W2SDataset


def represent_list(dumper: yaml.Dumper, data: dict):
    """Helper to save lists into yaml file"""
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", list(data), flow_style=True
    )


# Custom Dumper class to disable aliases
class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True  # This disables the automatic use of aliases (anchors)


def map_specimen(specimen: str):
    """maps the specimen names to the ones needed by AUTO-DIP

    Args:
        specimen (str): name of the specimen

    Returns:
        str: name of the specimen for AUTO-DIP
    """

    match specimen:
        case "F-actin":
            return "actin"
        case "actin":
            return "actin"
        case "mito":
            return "mito"
        case "nucleus":
            return "nucleus"
        case _:
            return None


def create_config_transfer(
    params_image: dict,
    exp_name: str,
    image_dict: dict,
    save_folder: Path,
    repr_img: str = None,
):
    """create a config for a run with the deep image prior with parameter transfer

    Args:
        params_image (dict): parameters describing input image
        exp_name (str): experiment name for MLflow
        image_dict (dict): dict containing info about the image (row in dataset
        dataframe)
        save_folder (Path): path to save to configs
        repr_img (str): type of representative image ("downsize" or "centercrop")
    """

    name = dataset.create_img_name(image_dict, repr_img=repr_img)

    params_save = {
        "verbosity": 0,
        "mlflow": True,
        "mlflow_log_loss_types": ["mse", "ssim", "psnr", "l1", "lpips"],
        "mlflow_exp_name": exp_name,
        "mlflow_run_name": name,
        "mlflow_log_images": False,
        "save_imgs": True,
        "save_path": str(Path("./search_results/") / exp_name_raw / "images_transfer"),
    }

    transfer_params = {}
    transfer_params["representative_img"] = repr_img
    transfer_params["microscope"] = image_dict["microscope_type"]
    transfer_params["specimen"] = map_specimen(image_dict["specimen"])

    params = {
        "image": params_image,
        "transfer_params": transfer_params,
        "save_and_log": params_save,
    }

    # Add the custom representation for lists
    yaml.add_representer(list, represent_list)

    save_path = save_folder / f"{name}.yaml"

    with save_path.open("w") as f:
        yaml.dump(params, f, Dumper=NoAliasDumper)


def create_config_orig(
    params_image: dict, exp_name: str, image_dict: dict, save_folder: Path
):
    """create a config for a run with the deep image prior with standard original
    parameters

    Args:
        params_image (dict): parameters describing input image
        exp_name (str): experiment name for MLflow
        image_dict (dict): dict containing info about the image (row in dataset
        dataframe)
        save_folder (Path): path to save to configs
    """

    name = dataset.create_img_name(image_dict)

    params_save = {
        "verbosity": 0,
        "mlflow": True,
        "mlflow_log_loss_types": ["mse", "ssim", "psnr", "l1", "lpips"],
        "mlflow_exp_name": exp_name,
        "mlflow_run_name": name,
        "mlflow_log_images": False,
        "save_imgs": True,
        "save_path": str(Path("./search_results/") / exp_name_raw / "images_orig"),
    }

    params_net = {
        "num_iter": 1800,
        "skip_n11": 4,
        "num_scales": 5,
        "skip_n33d": 128,
        "skip_n33u": 128,
    }

    params = {"image": params_image, "net": params_net, "save_and_log": params_save}

    # Add the custom representation for lists
    yaml.add_representer(list, represent_list)

    save_path = save_folder / f"{name}.yaml"

    with save_path.open("w") as f:
        yaml.dump(params, f, Dumper=NoAliasDumper)


if __name__ == "__main__":
    datasets = [
        HagenTestDataset(root_dir=Path("./data/hagen"), downsize_method="split"),
        FMDTestMixDataset(root_dir=Path("./data/fmd/test_mix/")),
        FMDDataset(
            root_dir=Path("./data/fmd/"),
            select_imgs_file=Path("./data/data_selection/fmd_calibration.csv"),
        ),
        ShahDataset(
            root_dir=Path("./data/shah/"),
            select_imgs_file=Path("./data/data_selection/shah_sel.csv"),
        ),
        BioSRDataset(
            root_dir=Path("./data/biosr"),
            select_imgs_file=Path("./data/data_selection/biosr_sel.csv"),
        ),
        W2SDataset(
            root_dir=Path("./data/w2s"),
            select_imgs_file=Path("./data/data_selection/w2s_sel.csv"),
        ),
    ]

    for dataset in datasets:
        config_path = Path("./configs/") / dataset.name

        exp_name_raw = dataset.name

        config_path_trans = config_path / "transfer"
        config_path_orig = config_path / "orig"

        if not config_path_trans.exists():
            os.makedirs(config_path_trans)
        if not config_path_orig.exists():
            os.makedirs(config_path_orig)

        sel_imgs_df = dataset.df
        if "image_quality" in sel_imgs_df.columns:
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

            create_config_transfer(
                params_image=params_image,
                exp_name=exp_name_raw + "_transfer",
                image_dict=img_row,
                save_folder=config_path_trans,
                repr_img="downsize",
            )
            create_config_orig(
                params_image=params_image,
                exp_name=exp_name_raw + "_orig",
                image_dict=img_row,
                save_folder=config_path_orig,
            )
