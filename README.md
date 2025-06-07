# FedVideoSeg: Federated Learning for Educational Video Segment Localization

FedVideoSeg is a privacy-preserving federated learning framework that combines three key technologies:
1. **Large Language Models** (DeepSeek) for semantic video segment evaluation
2. **Federated Learning** for collaborative training across heterogeneous devices
3. **Binary Search Video Localization** for efficient question-relevant segment finding

## 🔥 Key Features

- **Binary Search Localization**: Reduces video processing complexity from O(T) to O(log T)
- **Device-Aware Federated Learning**: Supports mobile, desktop, and server devices with adaptive compression
- **Privacy Protection**: Local data processing with differential privacy options
- **Multimodal Fusion**: Dynamic attention mechanism combining visual, audio, and text information
- **Real-time Performance**: <45ms inference latency across all device types

## 🏗️ Architecture


## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_qwen.py --batch_size 1
   ```

2. **Missing Video Files**
   ```bash
   # Check video directory
   ls dataset/video/
   ```

3. **Privacy Budget Exhausted**
   ```bash
   # Increase noise multiplier or reduce training rounds
   python federated_train.py
   ```


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@article{fedvideoseg2024,
  title={FedVideoSeg: Privacy-Preserving Federated Learning for Educational Video Segment Localization},
  author={Dawei Yuan},
  journal={Conference/Journal},
  year={2024}
}
```

## 🔗 Related Projects

- [FedML](https://github.com/FedML-AI/FedML) - Federated Learning Framework
- [Flower](https://github.com/adap/flower) - Federated Learning Platform
- [Opacus](https://github.com/pytorch/opacus) - Differential Privacy Library

---

