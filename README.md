**[English](README.md)** | **[中文](./docs/README.md)**

# DocAligned

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/DocsaidLab/DocAligned/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/DocAligned?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8+-aff.svg"></a>
</p>

## Introduction

This project is a visual system focused on the localization of documents in the image. Our primary aim for this system is to provide predictions of the four corners of documents. This feature is critically important in applications dealing with fintech, banking, and the shared economy, offering a reduction in errors and computational requirements for various image processing and text analysis tasks.

## Dataset

- **MIDV-500/MIDV-2019**
   - [**MIDV**](https://github.com/fcakyon/midv500)
   - MIDV-500 comprises 500 video clips of 50 different identity document types, including 17 ID cards, 14 passports, 13 driver's licenses, and 6 other identity documents from various countries. It is authentic and allows extensive research on various document analysis problems.
   - The MIDV-2019 dataset includes distorted and low-light images.

- **Indoor Scenes**
   - [**Indoor**](https://web.mit.edu/torralba/www/indoor.html)
   - This dataset contains 67 indoor categories, with a total of 15,620 images. The number of images varies by category, but each has at least 100 images. All images are in jpg format.

- **CORD v0**
   - [**CORD**](https://github.com/clovaai/cord)
   - Comprising thousands of Indonesian receipts, this dataset includes image text annotations for OCR and multi-layer semantic labels for parsing. It can be used for a variety of OCR and parsing tasks.

- **Docpool**
   - [**Dataset**](./data/docpool/)
   - We collected various text images from the internet for use in dynamic composite image technology as a training dataset.

## Dataset Preprocessing

1. **Install MIDV-500 Package:**

    ```bash
    pip install midv500
    ```

2. **Download Datasets:**

    - **MIDV-500/MIDV-2019:**
      After installation, execute `download_midv.py`.

      ```bash
      cd DocAligned/data
      python download_midv.py
      ```

    - **MIT Indoor Scenes & CORD v0:**
      Visit their respective links and follow the download instructions.

3. **Build Dataset:**

    Place MIDV and CORD datasets in the same location, and set the `ROOT` variable in `build_dataset.py` to the directory where datasets are stored. Then, execute:

    ```bash
    python build_dataset.py
    ```

   This process will generate several `.json` files containing all dataset information, including image paths, labels, image sizes, etc.

## Building the Training Environment

First, ensure you have built the base image `docsaid_training_base_image` from `DocsaidKit`. If not, refer to `DocsaidKit` documentation. Then, use the following command to build the Docker image for DocAligned work:

```bash
cd DocAligned
bash docker/build.bash
```

Our default [Dockerfile](./docker/Dockerfile) is specifically designed for document alignment training. You may modify it as needed. Here is an explanation of the file:

1. **Base Image**
    - `FROM docsaid_training_base_image:latest`
    - This line specifies the base image for the container, the latest version of `docsaid_training_base_image`. The base image is like a starting point for building your Docker container, containing a pre-configured operating system and some basic tools. You can find it in the `DocsaidKit` project.

2. **Working Directory**
    - `WORKDIR /code`
    - The container's working directory is set to `/code`. This is a directory in the Docker container where your application and all commands will operate.

3. **Environment Variable**
    - `ENV ENTRYPOINT_SCRIPT=/entrypoint.sh`
    - This defines an environment variable `ENTRYPOINT_SCRIPT` set to `/entrypoint.sh`. Environment variables store common configurations and can be accessed anywhere in the container.

4. **Install gosu**
    - `RUN` command installs `gosu`, a lightweight tool that allows users to execute commands with a specific user identity, similar to `sudo` but more suited for Docker containers.
    - `apt-get update && apt-get install -y gosu && rm -rf /var/lib/apt/lists/*` updates the package list, installs `gosu`, and then cleans up unnecessary files to reduce image size.

5. **Create Entry Point Script**


    - A series of `RUN` commands create the entry point script `/entrypoint.sh`.
    - The script first checks if environment variables `USER_ID` and `GROUP_ID` are set. If so, it creates a new user and group with the same IDs and runs commands as that user.
    - Useful for handling file permission issues inside and outside the container, especially when the container needs to access files on the host machine.

6. **Permission Assignment**
    - `RUN chmod +x "$ENTRYPOINT_SCRIPT"` makes the entry point script executable.

7. **Set Container Entry Point and Default Command**
    - `ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]` and `CMD ["bash"]`
    - These commands specify the default command executed when the container starts. When the container launches, it runs the `/entrypoint.sh` script.

## Running Training (Based on Docker)

This section explains how to use the Docker image you've built for document alignment training.

First, examine the contents of the `train.bash` file:

```bash
#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocAligned.model import main_docalign_train

if __name__ == '__main__':
    Fire(main_docalign_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligned:/code/DocAligned \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python trainer.py --cfg_name $1
```

Explanation:

1. **Create Training Script**
   - `cat > trainer.py <<EOF ... EOF`
   - This command creates a Python script `trainer.py`. This script imports necessary modules and functions and invokes `main_docalign_train` in the script's main section. It uses Google's Python Fire library for command-line argument parsing, making CLI generation easier.

2. **Run Docker Container**
   - `docker run ... doc_align_train python trainer.py --cfg_name $1`
   - This command starts a Docker container and runs the `trainer.py` script inside it.
   - `-e USER_ID=$(id -u) -e GROUP_ID=$(id -g)`: These parameters pass the current user's user ID and group ID to the container to create a user with corresponding permissions inside.
   - `--gpus all`: Specifies that the container can use all GPUs.
   - `--shm-size=64g`: Sets the size of shared memory, useful in large-scale data processing.
   - `--ipc=host --net=host`: These settings allow the container to use the host's IPC namespace and network stack, helping with performance.
   - `--cpuset-cpus="0-31"`: Specifies which CPU cores the container should use.
   - `-v $PWD/DocAligned:/code/DocAligned` etc.: These are mount parameters, mapping directories from the host to the container for easy access to training data and scripts.
   - `--cfg_name $1`: This is an argument passed to `trainer.py`, specifying the name of the configuration file.

3. **Dataset Path**
   - Note that `/data/Dataset` is the path for training data. Adjust `-v /data/Dataset:/data/Dataset` to match your dataset directory.

Finally, return to the `DocAligned` parent directory and execute the following command to start training:

```bash
bash DocAligned/docker/train.bash LC150_BIFPN64_D3_PointReg_r256 # Replace with your configuration file name
```

- Note: For configuration file details, refer to [DocAligned/model/README.md](./model/README.md).

By following these steps, you can safely execute document alignment training tasks within a Docker container, leveraging Docker's isolated environment for consistency and reproducibility. This approach makes project deployment and scaling more convenient and flexible.
