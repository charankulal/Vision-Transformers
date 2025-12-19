# Model Source & Attribution

## Origin

This implementation is based on the **Vision Transformer (ViT)** architecture introduced in the paper:

**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**
*Dosovitskiy et al., ICLR 2021*

📄 Paper: [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
🔗 Official Repository: [google-research/vision_transformer](https://github.com/google-research/vision_transformer)

## Implementation

This is a **clean-room TensorFlow/Keras implementation** following the architecture specifications from the original paper. All 8 standard variants (S/B/L/H with different patch sizes) are implemented from scratch based on the published specifications.

**Key Features:**
- ✅ Architecture faithful to original paper
- ✅ Built using TensorFlow 2.x and Keras
- ✅ No pre-trained weights included (architecture only)
- ✅ Designed for easy fine-tuning and transfer learning

## License

The Vision Transformer architecture is released under **Apache License 2.0** by Google Research.

This implementation code is provided for educational and research purposes.

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```

## What's Included

- ✅ **Architecture implementation**: All 8 ViT variants (S/16, S/32, B/16, B/32, L/16, L/32, H/14, H/16)
- ✅ **Modular components**: Patch embedding, multi-head attention, transformer blocks
- ✅ **Factory functions**: Easy creation of any variant
- ❌ **Pre-trained weights**: Not included (train from scratch or use your own)

## Acknowledgments

Special thanks to:
- **Google Research** for the original Vision Transformer research
- **Alexey Dosovitskiy** and the ViT team for the groundbreaking work
- The broader research community for advancing transformer architectures in computer vision
