# AUTO-DIP

## About

AUTO-DIP automates the parameter selection for Deep Image Prior denoising for microscopy images. It is based on Deep Image Prior ([DIP](https://github.com/DmitryUlyanov/deep-image-prior)) [1] and does not require any training data. This add-on for DIP chooses parameters automatically based on microscope type, specimen, and the image itself. The code is associated with our [paper](https://doi.org/10.48550/arXiv.2601.12055) [2].


## Usage

1. Clone the repo and initialize submodules by running 
    ```bash
    git clone --recurse-submodules https://github.com/IPMI-ICNS-UKE/DECO-DIP
    ```
    in the command line.

2. Apply patch files for the deep-image-prior repo:
    ```bash
    (cd dip && git apply ../dip.patch)
    ```

3. Set up the python virtual environment with uv:
    ```bash
    #install uv (if not already installed)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # install packages
    uv sync
    # activate environment
    source .venv/bin/activate
    ```

4. Run the program. With the parameter param_path you can specify a yaml file containing the parameters. Default is ./parameters.yaml.
    ```bash
    ./main.py --param_path parameters.yaml
    ```

    If you want to use MLflow for tracking, start the MLflow server before running AUTO-DIP:
    ```bash
    mlflow server --host 127.0.0.1 --port 50000
    ```

    If you want to run the program with more than one parameter file, you can specify a folder with parameter files. All yaml files in that folder are processed successively.

    Example config files can be found in [./example_configs](./example_configs) and default parameters are stored in [./default_parameters.yaml](./default_parameters.yaml).

    For detailed parameter descriptions see [./default_parameters.yaml](./default_parameters.yaml) and the [parameter description in the wiki](../../wiki/Parameter-Description).

## References

[1] Ulyanov D, Vedaldi A, Lempitsky V. Deep Image Prior. International Journal of Computer Vision 2020; 128 (7): 1867–88. https://doi.org/10.1007/s11263-020-01303-4.
[2] Meyer L, Wissel F, Knopp T, Pfefferle S, Fliegert R, Sandmann M, Uebler L, Möckl F, Diercks B-P, Lohr D, Werner R. Automating Parameter Selection in Deep Image Prior for Fluorescence
 Microscopy Image Denoising via Similarity-Based Parameter Transfer. https://doi.org/10.48550/arXiv.2601.12055. 
 
