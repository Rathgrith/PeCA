# PeCA

Official ECCV 2026 code release for **PeCA: Palette Context Assisted Inference for Test-Time Paint-Bucket Colourisation on Animation Videos**.

[🌐 **Project Page**](https://rathgrith.github.io/PeCA/) ·
[📄 **Paper**](https://arxiv.org/abs/2608.00903) ·
[🗂️ **Anita-Pirate Dataset**](https://www.kaggle.com/datasets/f4d74236ea8f91768f5c81236f84e86080cf26a4de0c0833ac760c354d0dc365)


[![PeCA framework overview](../assets/figures/overview.png)](https://rathgrith.github.io/PeCA/)

## Setup

```bash
conda env create -f environment.yml
conda activate peca
```

The default `environment.yml` follows the original HPC environment which used one NVIDIA A100-SXM4-40GB GPU, (while this release is tested on a local workstation with one NVIDIA GeForce RTX 5090 GPU). 

## Data

- **PBC-3D:** follow the
  [BasicPBC instructions](https://github.com/ykdai/BasicPBC#data-download).
- **PBC-Real:** request non-commercial evaluation access from the BasicPBC
  authors at the same link.
- **Anita-Pirate:** access [here](https://www.kaggle.com/datasets/f4d74236ea8f91768f5c81236f84e86080cf26a4de0c0833ac760c354d0dc365).

Default dataset paths are under `data/`. A custom path can be supplied with `--dataset-root`.

## Run

```bash
# Default: PBC-3D design sheet, SAM2.1-Large + PeCA.
bash run.sh

# Inspect the resolved configuration without loading data or a model.
bash run.sh --dry-run
```

The random seed is `null` by default, so ARE sampling and therefore the final results can vary between runs.

### Change the backbone

```bash
# Use DINOv2 ViT-L/14 while keeping the default dataset and protocol.
bash run.sh --backbone dinov2_vitl14
```

### Change the number of references

```bash
# Use five design-sheet references (the default protocol uses one).
bash run.sh --protocol design_sheet_5shot
```

Use `--protocol design_sheet_maxshot` to use every reference sheet available for each sequence.

Available backbone names are `sam2_1_large`, `dinov2_vitl14`, `dinov3_convnext_l`, `siglip2_vit_b16`, `stable_diffusion_2_1`, and `dacon_v1_1`.

Stable Diffusion additionally requires `pip install -e '.[sd]'`. The other training-free backbones are included in the default environment.

### Change the dataset & settings

```bash
# Anita-Pirate in-between colourisation with DINOv2 ViT-L/14.
bash run.sh \
  --dataset anita_pirate \
  --protocol inbetween_anita_pirate \
  --backbone dinov2_vitl14

# PBC-Real first-frame colourisation with SAM2.1-Large.
bash run.sh --dataset pbc_real --protocol first_frame
```

See more dataset, protocol, backbone, and method component files under `configs/`.


Achieving the best colourisation results requires an external pretrained model DACoN 1.1. 

```bash
# Download dacon_v1_1.pth manually from the DACoN repository, then:
bash run.sh \
  --backbone dacon_v1_1 \
  --checkpoint checkpoints/dacon_v1_1.pth
```

The checkpoint source and SHA-256 are stored in `configs/backbones/dacon_v1_1.yaml`. Run `bash run.sh --wandb` to log the resolved config and metrics. For an existing minimal environment, install `.[wandb]` first. Some reference runs are linked in the [PeCA W&B report](https://api.wandb.ai/links/505029658/9dcpgyo2).

## Citation

```bibtex
@inproceedings{lin2026peca,
title = "PeCA: Palette Context Assisted Inference for Test-Time Paint-Bucket Colourisation on Animation Videos",
author = "Dongheng Lin and Jianbo Jiao",
note = "The 19th European Conference on Computer Vision, ECCV 2026 ; Conference date: 08-09-2026 Through 12-09-2026",
year = "2026",
language = "English",
series = "Lecture Notes in Computer Science",
publisher = "Springer",
booktitle = "Computer Vision – ECCV 2026",
url = "https://eccv.ecva.net/Conferences/2026",
}
```
## Licenses & Acknowledgements

This release contains adapted evaluation and model-wrapper code originating from [DACoN](https://github.com/kzmngt/DACoN) (Thanks for their excellent work!), distributed under the MIT License. Similarly, PeCA's implementation and release infrastructure are provided under MIT License.

Other foundation-model implementations and weights are downloaded at runtime and are not redistributed here. They remain subject to their respective terms:

- [DINOv2](https://github.com/facebookresearch/dinov2)
- [SAM 2](https://github.com/facebookresearch/sam2) and the
  [SAM2.1-Large weights](https://huggingface.co/facebook/sam2.1-hiera-large)
- [timm](https://github.com/huggingface/pytorch-image-models), including the
  [DINOv3 ConvNeXT-L](https://huggingface.co/timm/convnext_large.dinov3_lvd1689m)
  and [SigLIPv2 ViT-B/16](https://huggingface.co/timm/vit_base_patch16_siglip_512.v2_webli)
  checkpoints
- [Stable Diffusion 2.1](https://huggingface.co/sd2-community/stable-diffusion-2-1)

For datasets, PBC-3D and PBC-Real are governed by the [BasicPBC repository and its data terms](https://github.com/ykdai/BasicPBC#data-download). In particular, PBC-Real requires a request to the original authors. Anita-Pirate is derived from the CC-BY-licensed [Anita Dataset](https://zhenglinpan.github.io/AnitaDataset_homepage/).
