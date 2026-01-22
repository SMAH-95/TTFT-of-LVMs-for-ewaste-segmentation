# An artificial-intelligence-driven framework for sustainable e-waste recognition

This repo contains the supported pytorch code and configurations for the "An artificial-intelligence-driven framework for sustainable e-waste recognition" article. 

## Abstract
Recycling e-waste is critical for resource conservation and environmental sustainability. Artificial-intelligence (AI) has shown strong potential for waste recycling. Yet e-waste recognition remains challenging because images are heterogeneous and cluttered, with overlapping components that create ambiguous boundaries and lead to suboptimal recognition. Here, we present an AI-driven framework that integrates test-time fine-tuning (TTFT) with large-scale vision foundation models to automate e-waste recognition in material recovery facilities (MRFs). We validate the framework on a curated e-waste dataset reflecting real-world conditions for automated resource recovery. In our experiments, TTFT is approximately 60× faster than training-time fine-tuning, supporting scalable adaptation while maintaining reliable segmentation. The framework achieves a frequency-weighted intersection-over-union of 0.38 for automatic segmentation, and 0.85 (box) and 0.67 (point) for promptable segmentation. Compared with training-time fine-tuning, our framework improves segmentation accuracy while requiring less data, supporting deployment in MRFs and contributing to the United Nations Sustainable Development Goals.

## System requirements
We implemented the proposed framework in PyTorch and ran experiments on a single NVIDIA A100 PCIE (40 GB).
