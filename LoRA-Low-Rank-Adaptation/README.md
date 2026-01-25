# LoRA: Low-Rank Adaptation of Large Language Models

## 📄 Paper Details
- **Title**: LoRA: Low-Rank Adaptation of Large Language Models
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
- **Organization**: Microsoft
- **Published**: ICLR 2022
- **Paper Link**: https://arxiv.org/abs/2106.09685

## 🎯 What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Freezes the pre-trained model weights
- Injects trainable low-rank decomposition matrices into each layer
- Reduces trainable parameters by up to 10,000x while maintaining performance
- Enables efficient task-specific adaptation without full model retraining

**Key Innovation**: Instead of fine-tuning all parameters, LoRA learns a low-rank "update" to the weight matrices:
```
W_new = W_frozen + B × A
```
where B and A are small trainable matrices with rank r << d.

## 📊 Implementation Details

This implementation includes:

1. **Custom LoRALayer Module**
   - Wraps any Linear layer with low-rank adaptation
   - Configurable rank (r) and scaling factor (alpha)
   - Proper weight initialization

2. **MNIST Fine-Tuning Example**
   - Pre-training on full MNIST dataset
   - Fine-tuning with LoRA on specific digit (digit 9)
   - Parameter efficiency demonstration (2.5% trainable parameters)

3. **Visualization Tools**
   - Weight matrix visualization
   - LoRA update visualization
   - Prediction confidence visualization

4. **Transformer Integration**
   - Self-Attention with LoRA injection
   - Q and V projection adaptation
   - Ready for text/sequence tasks

5. **Experimental Analysis**
   - Rank efficiency experiments (r=1, 4, 64)
   - Performance vs. parameter trade-offs

## 🚀 Usage

### Basic LoRA Layer
```python
from torch import nn

# Original layer
linear = nn.Linear(1000, 500)

# Wrap with LoRA
lora_linear = LoRALayer(linear, r=8, alpha=16)

# Only LoRA parameters are trainable
# Original weights are frozen
```

### Fine-Tuning Example
```python
# 1. Freeze base model
for param in model.parameters():
    param.requires_grad = False

# 2. Inject LoRA
model.layer = LoRALayer(model.layer, r=8, alpha=16)

# 3. Train only LoRA parameters
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))
```

## 📈 Results

- **Parameter Efficiency**: 2.5% of original parameters (32K vs 1.3M)
- **Accuracy on Digit 9**: Maintains high accuracy with minimal parameters
- **Training Speed**: Faster convergence due to fewer parameters

## 🔬 Key Concepts Demonstrated

1. **Low-Rank Decomposition**: W = B @ A where rank(B @ A) = r
2. **Parameter Freezing**: Original weights remain unchanged
3. **Efficient Adaptation**: Only small matrices are learned
4. **Merge/Unmerge**: LoRA weights can be merged into base model for deployment
5. **Multi-task Learning**: Different LoRA adapters for different tasks

## 📋 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.5.0
tqdm>=4.65.0
```

## 🎓 Learning Outcomes

After studying this implementation, you'll understand:
- How LoRA reduces memory and compute requirements
- When and why to use parameter-efficient fine-tuning
- How to inject LoRA into any Linear layer
- The trade-off between rank and model capacity
- How LoRA applies to Transformers and attention mechanisms

## 📚 References

```bibtex
@inproceedings{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add more examples

---

**Implementation by**: Satyam Mistari  
**Date**: January 2026
