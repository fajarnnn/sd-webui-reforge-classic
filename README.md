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

<details>
<summary>(Unscientific) Comparisons</summary>

<table align="center">
<thead>
    <tr align="center">
        <th></th>
        <th>Forge Classic</th>
        <th>Forge <code>previous</code></th>
        <th>Forge <code>main</code></th>
        <th>reForge <code>main</code></th>
    </tr>
</thead>
<tbody>
    <tr>
        <td align="left"><sup>1</sup> Size</td>
        <td align="center">4.3 MB</td>
        <td align="center">6.8 MB</td>
        <td align="center"><sup>2</sup> 18.5 MB</td>
        <td align="center">7.8 MB</td>
    </tr>
    <tr>
        <td align="left"><sup>3</sup> Startup</td>
        <td align="center">4.5s</td>
        <td align="center"><sup>4</sup> 9.5s</td>
        <td align="center">5.2s</td>
        <td align="center">5.7s</td>
    </tr>
</tbody>
</table>

> **1:** using the <code>Download ZIP</code> button on GitHub<br>
> **2:** the large size is from `backend/huggingface`<br>
> **3:** using only `--xformers` flag; disable all **extra** extensions; does **not** include `import torch` time<br>
> **4:** the long time is from requirement conflicts

</details>

<hr>

## Features [Mar. 11]
> Most base features of the original [Automatic1111 Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) should still function

#### New Features

- [X] Support [uv](https://github.com/astral-sh/uv) package manager
    - requires **uv**
    - see [Commandline](#commandline)
- [X] Support [SageAttention](https://github.com/thu-ml/SageAttention)
    - requires RTX **30** +
    - ~5% speed up; only supports **SDXL**
    - see [Commandline](#commandline)
- [X] Support fast `cublas` operation *(`CublasLinear`)*
    - requires **manually** installing [cublas_ops](https://github.com/aredden/torch-cublas-hgemm) package
    - ~25% speed up
- [X] Support fast `fp8` operation *(`torch._scaled_mm`)*
    - requires RTX **40** +
    - ~10% speed up; reduce quality

> [!Note]
> The `cublas_ops` requires `fp16` precision, thus is not compatible with `fp8` settings

- [X] Support `v-pred` **SDXL** checkpoints *(**eg.** [NoobAI](https://civitai.com/models/833294?modelVersionId=1190596))*
- [X] Implement RescaleCFG
    - reduce burnt colors; mainly for `v-pred`
- [X] Implement `diskcache`
    - *(backported from Automatic1111 Webui upstream)*
- [X] Implement `skip_early_cond`
    - *(backported from Automatic1111 Webui upstream)*
- [X] Update `spandrel`
    - support most modern Upscaler architecture
- [X] Add `pillow-heif` package
    - support `.avif` and `.heif` formats
- [X] Add an option to disable **Refiner**
- [X] Add an option to disable ExtraNetworks **Tree View**
- [X] Support [Union](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0) / [ProMax](https://huggingface.co/brad-twinkl/controlnet-union-sdxl-1.0-promax) ControlNet
    - I just made them always show up in the dropdown

#### Removed Features

- [ ] SD2
- [X] Alt-Diffusion
- [X] Instruct-Pix2Pix
- [X] Hypernetworks
- [X] SVD
- [X] Z123
- [X] CLIP Interrogator
- [X] Deepbooru Interrogator
- [X] Textual Inversion Training
- [X] Checkpoint Merging
- [X] Most **built-in** Extensions
- [X] Some **built-in** Scripts
- [X] The `test` scripts
- [X] `Photopea` and `openpose_editor` *(ControlNet)*

#### Optimizations

- [X] Fix Memory Leak when switching Checkpoints
- [X] Fix `pydantic` Errors
- [X] Check for Extension Updates in Parallel
- [X] Clean up the `ldm_patched` *(**ie.** `comfy`)* folder
- [X] Remove unused `cmd_args`
- [X] Remove unused `shared_options`
- [X] Remove unused `args_parser`
- [X] Remove large amount of legacy code
- [X] Remove duplicated upscaler codes
    - put every upscaler inside the `ESRGAN` folder
- [X] Improve code logics
- [X] Improve hash caching
- [X] Improve error logs
    - no longer prints `TypeError: 'NoneType' object is not iterable`
- [X] Moved `embeddings` folder into `models` folder
- [X] ControlNet Rewrite
    - change Units to `gr.Tab`
    - remove multi-inputs, as they are "[misleading](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)"
    - change `visible` toggle to `interactive` toggle; now the UI will no longer jump around
    - improved `Presets` application
- [X] Lint & Format most of the Python and JavaScript codes
- [X] Update to latest PyTorch
    - currently `2.6.0+cu126`
- [X] Run `Clip` on CPU by default
- [X] Update recommended Python to `3.11.9`
- [X] `use_checkpoint: False`
- [X] many more... :tm:

<br>

## Commandline
> These flags can be added after the `set COMMANDLINE_ARGS=` line in the `webui-user.bat` *(separate each flag with space)*

#### A1111 built-in

- `--no-download-sd-model`: Do not download a default checkpoint
    - can be removed after you download some checkpoints of your choice
- `--xformers`: Install the `xformers` package to speed up generation
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
- `--sage`: Install the `sageattention` package to speed up generation
    - requires RTX **30** +
    - requires manually installing **triton**
    - only affects **SDXL**

> [!Tip]
> `--xformers` is still recommended even if you already have `--sage`, as `sageattention` does not speed up **VAE** while `xformers` does

- `--model-ref`: Points to a central `models` folder that contains all your models
    - said folder should contain subfolders like `Stable-diffusion`, `Lora`, `VAE`, `ESRGAN`, etc.

> [!Important]
> This simply **replaces** the `models` folder, rather than adding on top of it

<hr>

## Installation

1. Install **git**
    - https://git-scm.com/downloads
2. Install **Python**
    - (Recommended) Install **[uv](https://github.com/astral-sh/uv)**
    - (Manual) Install **[Python 3.11.9](https://www.python.org/downloads/release/python-3119/)**
3. Clone the Repo
    ```bash
    git clone https://github.com/Haoming02/sd-webui-forge-classic
    ```
4. Prepare **uv** *(if you installed it)*
    1. Set up **venv**
        ```bash
        cd sd-webui-forge-classic
        uv venv venv --python 3.11
        ```
    2. Add the `--uv` flag *(see [Commandline](#by-classic))*
5. Launch the Webui via `webui-user.bat`
6. On first launch, it will automatically install all the requirements
7. Once installation is finished, the Webui will start in a browser automatically

<hr>

<details>
<summary>GitHub Related</summary>

- **Issues** about removed features will simply be ignored; **Issues** regarding installation will also be ignored if it's obviously user-error
- **Feature Request** not related to performance or optimization will simply be ignored
    - For cutting edge features, use [reForge](https://github.com/Panchovix/stable-diffusion-webui-reForge) instead

</details>

<hr>

<p align="center">
Special thanks to <b>AUTOMATIC1111</b>, <b>lllyasviel</b>, and <b>comfyanonymous</b>, <b>kijai</b>, <br>
along with the rest of the contributors, <br>
for their invaluable efforts in the open-source image generation community
</p>
