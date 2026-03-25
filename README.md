# Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation

This repo contains the supported pytorch code and configurations for the "Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation" article. 

## Abstract
Recycling e-waste is critical for resource conservation and environmental sustainability. Artificial-intelligence (AI) has shown potential for waste recycling. Yet AI-based e-waste recognition remains challenging because images are heterogeneous and cluttered, while domain-specific annotated data are limited and conventional adaptation is time and compute-intensive. Here, we present an AI-driven method that integrates test-time fine-tuning (TTFT) with a large-scale vision foundation model to enable automated e-waste segmentation for sustainable resource recovery workflows. The method adapts the model at test-time using a small, diverse, contextually relevant retrieved set, reducing dependence on large labeled datasets and expensive fine-tuning. We evaluate the method on a curated e-waste dataset. In our experimental setting, end-to-end TTFT is approximately 6× faster than parameter-efficient fine-tuning while maintaining reliable segmentation and achieving higher accuracy with less domain-specific data, enabling scalable AI deployment for e-waste management and supporting progress toward the United Nations Sustainable Development Goals.

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
This research is supported by the [Automation and Sustainability in Construction and Intelligent Infrastructure (ASCII) Lab](https://www.monash.edu/ascii/home) at Monash University, Melbourne, Australia. The authors sincerely thank the lab for providing resources, technical assistance, and a collaborative setting that made this work possible.



This work was inspired by and builds on the following research works:

[Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08020)

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)



