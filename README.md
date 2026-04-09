# Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation

This repo contains the supported pytorch code and configurations for the "Advancing Resource Recovery from E-Waste with Artificial-Intelligence-Driven Segmentation" article. 

## Abstract
E-waste recycling is crucial for resource conservation and environmental sustainability. Artificial intelligence (AI) has shown potential to advance waste recognition and recycling. However, AI-based e-waste recognition remains challenging due to heterogeneous, cluttered images and limited labeled data. Furthermore, deep learning model adaptation methods for e-waste recognition are often constrained by high computational costs and lengthy adaptation times. Here, the study presents an AI-driven framework that integrates test-time fine-tuning (TTFT) with large-scale vision foundation models for automated e-waste segmentation, thereby supporting sustainable resource recovery workflows. TTFT adapts the model at inference time using a small, diverse, contextually relevant support set, reducing dependence on large, labeled datasets and expensive fine-tuning. A specialised dataset is curated to capture the complexity and material heterogeneity of e-waste across five key classes: lamps, cables, photovoltaic panels, ducts, and outlets. Evaluated on the curated e-waste dataset, the proposed method reduces end-to-end processing time by approximately 6× compared to parameter-efficient fine-tuning, while maintaining reliable segmentation and higher accuracy with less domain-specific data. This scalable approach advances AI deployment for e-waste management, supports progress toward the United Nations Sustainable Development Goals. 

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
This research is supported by the [Automation and Sustainability in Construction and Intelligent Infrastructure (ASCII) Lab](https://www.monash.edu/ascii) at Monash University, Melbourne, Australia. The authors gratefully acknowledge the lab’s resources, technical support, and collaborative environment that enabled the development of the research.

This work was inspired by and builds on the following research works:

[Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08020)

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)

## Citation
If you find our work valuable for your research, we kindly ask you to consider citing it.

[Senanayake, A. and Arashpour, M., 2025. Automated Electro-construction waste Sorting: Computer vision for part-level segmentation. Waste Management, 203, p.114883.](https://www.sciencedirect.com/science/article/pii/S0956053X25002946)

[Senanayake, A., Gautam, B., Harandi, M. and Arashpour, M., 2025. Sustainable resource management in construction: Computer vision for recognition of electro-construction waste (ECW). Resources, Conservation and Recycling, 221, p.108380.](https://www.sciencedirect.com/science/article/pii/S0921344925002599)

