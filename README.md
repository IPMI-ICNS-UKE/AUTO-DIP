# AUTO-DIP

## About

AUTO-DIP automates the parameter selection for Deep Image Prior denoising for microscopy images. It is based on Deep Image Prior ([DIP](https://github.com/DmitryUlyanov/deep-image-prior)) [1] and does not require any training data. This add-on for DIP chooses parameters automatically based on microscope type, specimen and the image itself.


## Usage

1. Clone the repo and initialize submodules by running 
    ```bash
    git clone --recurse-submodules https://github.com/lin17a/DECO-DIP
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
    uv sync
    source .venv/bin/activate
    ```

4. Run the program. With the parameter param_path you can specify a yaml file containing the parameters. Default is ./parameters.yaml.
    ```
    ./main.py --param_path parameters.yaml
    ```

    If you want to run the program with more than one parameter file, you can specify a folder with parameter files. All yaml files in that folder are processed successively.

    Example config files can be found in [./example_configs](./example_configs) and default parameters are stored in [./default_parameters.yaml](./default_parameters.yaml).

    For detailed parameter descriptions see [./default_parameters.yaml](./default_parameters.yaml) and [./wiki/parameter_description.md](./wiki/parameter_description.md).

## References

[1] Ulyanov, D., Vedaldi, A., & Lempitsky, V. 2020. "Deep Image Prior". International Journal of Computer Vision 128 (7): 1867–88. https://doi.org/10.1007/s11263-020-01303-4.

