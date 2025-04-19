# DETR Object Detection on AU-AIR Dataset: Experimental Analysis

## Introduction

This report presents an experimental analysis of the DETR (DEtection TRansformer) model for object detection on the AU-AIR dataset. We explore three different configurations to understand how various parameters affect the model's performance. The AU-AIR dataset contains aerial images with 8 object classes: Human, Car, Truck, Van, Motorbike, Bicycle, Bus, and Trailer.

## Experimental Setup

We conducted three experiments with different configurations:

1. **Baseline**: DETR with ResNet-50 backbone, pretrained on COCO dataset, fine-tuned on AU-AIR with default parameters
2. **Experiment 1**: Increased image resolution (1024×1024 vs 800×800 in baseline)
3. **Experiment 2**: Different learning rate schedule (cosine annealing vs step decay in baseline)

All experiments were conducted with the following common settings:
- Batch size: 4
- Optimizer: AdamW
- Base learning rate: 1e-4
- Weight decay: 1e-4
- Training epochs: 10

## Implementation Details

For each experiment, we implemented:
- Data loading and preprocessing specific to the AU-AIR dataset
- Model training with appropriate loss functions (classification, bounding box, and GIoU losses)
- Evaluation metrics including mAP, per-class AP, confusion matrices, and accuracy scores
- Visualization of predictions and training progress

## Results and Analysis

### Quantitative Results

| Metric | Baseline | Experiment 1 | Experiment 2 |
|--------|----------|--------------|--------------|
| mAP    | 0.XX     | 0.XX         | 0.XX         |
| Human AP | 0.XX   | 0.XX         | 0.XX         |
| Car AP | 0.XX     | 0.XX         | 0.XX         |
| Truck AP | 0.XX   | 0.XX         | 0.XX         |
| Van AP | 0.XX     | 0.XX         | 0.XX         |
| Motorbike AP | 0.XX | 0.XX       | 0.XX         |
| Bicycle AP | 0.XX | 0.XX         | 0.XX         |
| Bus AP | 0.XX     | 0.XX         | 0.XX         |
| Trailer AP | 0.XX | 0.XX         | 0.XX         |
| Accuracy | 0.XX   | 0.XX         | 0.XX         |

### Loss Curves

[Loss curves will be inserted here after experiments]

### Confusion Matrices

[Confusion matrices will be inserted here after experiments]

### Comparison with Baselines

We compare our results with the baseline models reported in the AU-AIR paper:

| Model | mAP |
|-------|-----|
| YOLOV3-Tiny | 0.3023 |
| MobileNetV2-SSDLite | 0.1950 |
| Our DETR (Baseline) | 0.XX |
| Our DETR (Experiment 1) | 0.XX |
| Our DETR (Experiment 2) | 0.XX |

## Discussion

### Effect of Image Resolution

[Analysis of how increased resolution affected performance]

### Effect of Learning Rate Schedule

[Analysis of how different learning rate schedules affected performance]

### Class-wise Performance Analysis

[Analysis of which classes performed better/worse and potential reasons]

### Comparison with State-of-the-Art

[How our implementation compares with existing methods]

## Conclusion

[Summary of findings and recommendations for future work]

## References

1. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., & Zagoruyko, S. (2020). End-to-End Object Detection with Transformers. In European Conference on Computer Vision (pp. 213-229).
2. Bozcan, I., & Kayacan, E. (2020). AU-AIR: A Multi-modal Unmanned Aerial Vehicle Dataset for Low Altitude Traffic Surveillance. In IEEE International Conference on Robotics and Automation (ICRA).
