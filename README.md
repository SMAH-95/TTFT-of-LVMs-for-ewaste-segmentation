# An artificial-intelligence-driven framework for sustainable e-waste recognition

This repo contains the supported pytorch code and configurations for the "An artificial-intelligence-driven framework for sustainable e-waste recognition" article. 

## Abstract
Recycling e-waste is critical for resource conservation and environmental sustainability. Artificial-intelligence (AI) has shown strong potential for waste recycling. Yet e-waste recognition remains challenging because images are heterogeneous and cluttered, with overlapping components that create ambiguous boundaries and lead to suboptimal recognition. Here, we present an AI-driven framework that integrates test-time fine-tuning (TTFT) with large-scale vision foundation models to automate e-waste recognition in material recovery facilities (MRFs). We validate the framework on a curated e-waste dataset reflecting real-world conditions for automated resource recovery. In our experiments, TTFT is approximately 60× faster than training-time fine-tuning, supporting scalable adaptation while maintaining reliable segmentation. The framework achieves a frequency-weighted intersection-over-union of 0.38 for automatic segmentation, and 0.85 (box) and 0.67 (point) for promptable segmentation. Compared with training-time fine-tuning, our framework improves segmentation accuracy while requiring less data, supporting deployment in MRFs and contributing to the United Nations Sustainable Development Goals.
## AI-driven framework

## System requirements
We implemented the proposed framework in PyTorch and ran experiments on a single NVIDIA A100 PCIE (40 GB). You can install all the requirement via:

```bash
pip install -r requirements.txt
```
## Quick strat
1. Download the dataset and split into query images (for testing) and candidate images ( for TTFT)
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


## Cite
If you find our work valuable for your research, we kindly ask you to consider citing it.

```bibtex
@article{senanayake2025automated,
  title={Automated Electro-construction waste Sorting: Computer vision for part-level segmentation},
  author={Senanayake, Aseni and Arashpour, Mehrdad},
  journal={Waste Management},
  volume={203},
  pages={114883},
  year={2025},
  publisher={Elsevier}
}
@article{senanayake2025sustainable,
  title={Sustainable resource management in construction: Computer vision for recognition of electro-construction waste (ECW)},
  author={Senanayake, Aseni and Gautam, Birat and Harandi, Mehrtash and Arashpour, Mehrdad},
  journal={Resources, Conservation and Recycling},
  volume={221},
  pages={108380},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement
This research is supported by the [Automation and Sustainability in Construction and Intelligent Infrastructure (ASCII) Lab](https://www.monash.edu/ascii/home) at Monash University, Melbourne, Australia. The authors sincerely thank the lab for providing resources, technical assistance, and a collaborative setting that made this work possible.

This work builds on and was inspired by the following research:

[Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs](https://arxiv.org/abs/2410.08020)

[SAM 2: Segment Anything in Images and Videos](https://github.com/facebookresearch/sam2)



