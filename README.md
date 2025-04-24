<h1 align="center">Stable Diffusion WebUI Forge - Classic</h1>

<p align="center"><img src="html\ui.webp" width=512 alt="UI"></p>

<blockquote><i>
<b>Stable Diffusion WebUI Forge</b> is a platform on top of the original <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">Stable Diffusion WebUI</a> by <ins>AUTOMATIC1111</ins>, to make development easier, optimize resource management, speed up inference, and study experimental features.<br>
The name "Forge" is inspired by "Minecraft Forge". This project aims to become the Forge of Stable Diffusion WebUI.<br>
<p align="right">- <b>lllyasviel</b><br>
<sup>(paraphrased)</sup></p>
</i></blockquote>

<br>

"**Classic**" mainly serves as an archive for the "`previous`" version of Forge, which was built on [Gradio](https://github.com/gradio-app/gradio) `3.41.2` before the major changes *(see the original [announcement](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/801))* were introduced. Additionally, this fork is focused exclusively on **SD1** and **SDXL** checkpoints, having various optimizations implemented, with the main goal of being the lightest WebUI without any bloatwares.

> [Installation](#installation)

<hr>

## Features [Apr. 24]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support `v-pred` **SDXL** checkpoints *(**eg.** [NoobAI](https://civitai.com/models/833294?modelVersionId=1190596))*
- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **uv**
    - drastically speed up installation
    - see [Commandline](#by-classic)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention)
    - requires **manually** installing the [triton](https://github.com/triton-lang/triton) package
        - [how to install](#install-triton)
    - requires RTX **30** +
    - ~10% speed up
    - see [Commandline](#by-classic)
- [X] Support fast `fp16_accumulation`
    - requires PyTorch **2.7.0** +
    - ~25% speed up
    - see [Commandline](#by-classic)
- [X] Support fast `cublas` operation *(`CublasLinear`)*
    - requires **manually** installing the [cublas_ops](https://github.com/aredden/torch-cublas-hgemm) package
        - [how to install](#install-cublas)
    - ~25% speed up
    - enable in **Settings**
- [X] Support fast `fp8` operation *(`torch._scaled_mm`)*
    - requires RTX **40** +
    - ~10% speed up; reduce quality
    - enable in **Settings**

> [!Note]
> - The `fp16_accumulation` and `cublas` operation achieve the same speed up; if you already install/update to `torch==2.7.0`, you do not need to go for `cublas_ops`
> - The `fp16_accumulation` and `cublas` operation require `fp16` precision, thus is not compatible with the `fp8` operation

- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred` checkpoints
- [X] Implement MaHiRo
    - alternative CFG calculation
    - [graph](https://www.desmos.com/calculator/wcztf0ktiq)
- [X] Implement `diskcache`
    - *(backported from Automatic1111 Webui upstream)*
- [X] Implement `skip_early_cond`
    - *(backported from Automatic1111 Webui upstream)*
- [X] Update `spandrel`
    - support most modern Upscaler architecture
- [X] Add `pillow-heif` package
    - support `.avif` and `.heif` formats
- [X] Automatic row split for `X/Y/Z Plot`
- [X] Add an option to disable **Refiner**
- [X] Add an option to disable ExtraNetworks **Tree View**
- [X] Support [Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) / [ProMax](https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax) ControlNet
    - I just made them always show up in the dropdown

#### Removed Features

- [X] SD2
- [X] Alt-Diffusion
- [X] Instruct-Pix2Pix
- [X] Hypernetworks
- [X] SVD
- [X] Z123
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Checkpoint Merging
- [X] LDSR
- [X] Most **built-in** Extensions
- [X] Some **built-in** Scripts
- [X] The `test` scripts
- [X] `Photopea` and `openpose_editor` *(ControlNet)*
- [X] Unix `.sh` launch scripts
    - You can still use this WebUI by copying a launch script from another working WebUI; I just don't want to maintain them...

#### Optimizations

- [X] **[Freedom]** Natively integrate the `SD1` and `SDXL` logics
    - no longer `git` `clone` any repository on fresh install
    - no more random hacks and monkey patches
- [X] Fix memory leak when switching checkpoints
- [X] Clean up the `ldm_patched` *(**ie.** `comfy`)* folder
- [X] Remove unused `cmd_args`
- [X] Remove unused `shared_options`
- [X] Remove unused `args_parser`
- [X] Remove legacy codes
- [X] Remove duplicated upscaler codes
    - put every upscaler inside the `ESRGAN` folder
- [X] Improve color correction
- [X] Improve code logics
- [X] Improve hash caching
- [X] Improve error logs
    - no longer prints `TypeError: 'NoneType' object is not iterable`
- [X] Improve setting descriptions
- [X] Check for Extension updates in parallel
- [X] Moved `embeddings` folder into `models` folder
- [X] ControlNet Rewrite
    - change Units to `gr.Tab`
    - remove multi-inputs, as they are "[misleading](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)"
    - change `visible` toggle to `interactive` toggle; now the UI will no longer jump around
    - improved `Presets` application
- [X] Run `text encoder` on CPU by default
- [X] Fix `pydantic` Errors
- [X] Fix `Soft Inpainting`
- [X] Lint & Format most of the Python and JavaScript codes
- [X] Update to Pillow 11
    - faster image processing
- [X] Update `protobuf`
    - faster `insightface` loading
- [X] Update to latest PyTorch
    - currently `2.7.0+cu128`
- [X] No longer install `open-clip` twice
- [X] Update certain packages to newer versions
- [X] Update recommended Python to `3.11.9`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(separate each flag with space)*

#### A1111 built-in

- `--no-download-sd-model`: Do not download a default checkpoint
    - can be removed after you download some checkpoints of your choice
- `--xformers`: Install the `xformers` package to speed up generation
    - Currently, `torch==2.7.0` does **not** support `xformers` yet
- `--port`: Specify a server port to use
    - defaults to `7860`
- `--api`: Enable [API](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) access

<br>

- Once you have successfully launched the WebUI, you can add the following flags to bypass some validation steps in order to improve the Startup time
    - `--skip-prepare-environment`
    - `--skip-install`
    - `--skip-python-version-check`
    - `--skip-torch-cuda-test`
    - `--skip-version-check`

> [!Important]
> Remove them if you are installing an Extension, as those also block Extension from installing requirements

#### by. Forge

- For RTX **30** and above, you can add the following flags to slightly increase the performance; but in rare occurrences, they may cause `OutOfMemory` errors or even crash the WebUI; and in certain configurations, they may even lower the speed instead
    - `--cuda-malloc`
    - `--cuda-stream`
    - `--pin-shared-memory`

#### by. Classic

- `--uv`: Replace the `python -m pip` calls with `uv pip` to massively speed up package installation
    - requires **uv** to be installed first *(see [Installation](#installation))*
- `--uv-symlink`: Same as above; but additionally pass `--link-mode symlink` to the commands
    - significantly reduces installation size (`~7 GB` to `~100 MB`)

> [!Important]
> Using `symlink` means it will directly access the packages from the cache folders; refrain from clearing the cache when setting this option

- `--fast-fp16`: Enable the `allow_fp16_accumulation` option
    - requires PyTorch **2.7.0** +
- `--sage`: Install the `sageattention` package to speed up generation
    - requires **triton**
    - requires RTX **30** +
    - only affects **SDXL**

> [!Tip]
> `--xformers` is still recommended even if you already have `--sage`, as `sageattention` does not speed up **VAE** while `xformers` does

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder, rather than adding on top of it

<hr>

## Installation

0. Install **[git](https://git-scm.com/downloads)**
1. Clone the Repo
    ```bash
    git clone https://github.com/Haoming02/sd-webui-forge-classic
    ```

2. Setup Python

<details>
<summary>Recommended Method</summary>

- Install **[uv](https://github.com/astral-sh/uv)**
- Set up **venv**
    ```bash
    cd sd-webui-forge-classic
    uv venv venv --python 3.11 --seed
    ```
- Add the `--uv` flag to `webui-user.bat`

</details>

<details>
<summary>Standard Method</summary>

- Install **[Python 3.11.9](https://www.python.org/downloads/release/python-3119/)**
    - Remember to enable `Add Python to PATH`

</details>

3. **(Optional)** Configure [Commandline](#commandline)
4. Launch the WebUI via `webui-user.bat`
5. During the first launch, it will automatically install all the requirements
6. Once the installation is finished, the WebUI will start in a browser automatically

<br>

### Install cublas

<details>
<summary>Expand</summary>

0. Ensure the WebUI can properly launch already, by following the [installation](#installation) steps first
1. Open the console in the WebUI directory
    ```bash
    cd sd-webui-forge-classic
    ```
2. Start the virtual environment
    ```bash
    venv\scripts\activate
    ```
3. Create a new folder
    ```bash
    mkdir repo
    cd repo
    ```
4. Clone the repo
    ```bash
    git clone https://github.com/aredden/torch-cublas-hgemm
    cd torch-cublas-hgemm
    ```
5. Install the library
    ```
    pip install -e . --no-build-isolation
    ```

    - If you installed `uv`, use `uv pip install` instead
    - The installation takes a few minutes

</details>

### Install triton

<details>
<summary>Expand</summary>

0. Ensure the WebUI can properly launch already, by following the [installation](#installation) steps first
1. Open the console in the WebUI directory
    ```bash
    cd sd-webui-forge-classic
    ```
2. Start the virtual environment
    ```bash
    venv\scripts\activate
    ```
3. Install the library
    - **Windows**
        ```bash
        pip install triton-windows
        ```
    - **Linux**
        ```bash
        pip install triton
        ```
    - If you installed `uv`, use `uv pip install` instead

</details>

### Install sageattention 2
> If you only use **SDXL**, then `1.x` is already enough; `2.x` simply has partial support for **SD1** checkpoints

<details>
<summary>Expand</summary>

0. Ensure the WebUI can properly launch already, by following the [installation](#installation) steps first
1. Open the console in the WebUI directory
    ```bash
    cd sd-webui-forge-classic
    ```
2. Start the virtual environment
    ```bash
    venv\scripts\activate
    ```
3. Create a new folder
    ```bash
    mkdir repo
    cd repo
    ```
4. Clone the repo
    ```bash
    git clone https://github.com/thu-ml/SageAttention
    cd SageAttention
    ```
5. Install the library
    ```
    pip install -e . --no-build-isolation
    ```

    - If you installed `uv`, use `uv pip install` instead
    - The installation takes a few minutes

</details>

<br>

### Install older PyTorch
> Read this if your GPU does not support the latest PyTorch

<details>
<summary>Expand</summary>

0. Navigate to the WebUI directory
1. Edit the `webui-user.bat` file
2. Add a new line to specify an older version:
```bash
set TORCH_COMMAND=pip install torch==2.1.2 torchvision==0.16.2 --extra-index-url https://download.pytorch.org/whl/cu121
```

</details>

<hr>

### GitHub Related

- **Issues** about removed features will simply be ignored
- **Issues** regarding installation will be ignored if it's obviously user-error
- **Feature Request** not related to performance or optimization will simply be ignored
    - For cutting edge features, check out [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) instead

</details>

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>
