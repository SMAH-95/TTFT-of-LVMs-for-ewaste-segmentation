# Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation

This repo contains the supported pytorch code and configurations for the "Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation" article. 

## Abstract
Large-scale vision foundation models (LVMs) have demonstrated remarkable adaptability in computer vision. However, their direct application to electronic waste (e-waste) imaging is constrained by visual heterogeneity, scene clutter, and annotation scarcity, hindering effective generalization to fine-grained segmentation tasks. Here we present a deep learning–based segmentation approach that integrates test-time fine-tuning (TTFT) with LVMs for e-waste recognition, a domain defined by high intra-class variability and limited labeled data. TTFT adapts the model at inference time using a small, diverse, contextually relevant support set, eliminating the need for offline fine-tuning. To construct this support set, we introduce "active retrieval", a mechanism that couples cosine similarity selection with an informativeness acquisition function to select query-specific, non-redundant support images. Evaluated on a curated dataset, TTFT outperforms parameter-efficient fine-tuning across all segmentation settings while reducing end-to-end processing time approximately 6×. These results establish a deployable paradigm for lightweight, data-scarce LVM specialization.

## AI-driven segmention method
<img width="1158" height="620" alt="image" src="https://github.com/user-attachments/assets/2ce6a61c-ab7b-487e-98f4-afdce9913a5d" />

## System requirements
We implemented the proposed framework in PyTorch and ran experiments on a single NVIDIA A100 PCIE (40 GB). You can install all the requirement via:

```bash
pip install -r requirements.txt
```
## Quick strat
1. Download the dataset and split into query images (for testing) and candidate images (for TTFT)
2. Download the pre-trained [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/sam2) save in ./checkpoints
3. Download the [activeft](https://github.com/jonhue/activeft)
4. Image retrieval and TTFT:

Automatic segmenetaion
```bash
python TTFT_automatic_seg.py
```

Box propmt segmenetaion
```bash
python TTFT_box_seg.py
```

Point propmt segmenetaion
```bash
python TTFT_point_seg.py
```

## Acknowledgement

This work was inspired by and builds on the following research works:

[Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08020)

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)
