# Llama-3.2-3B Fine-Tuning with Unsloth

This Jupyter notebook demonstrates how to fine-tune the **Llama-3.2-3B-Instruct** model using the **Unsloth** library and the **FineTome-100k** dataset. The notebook includes model training, inference, and an interactive Gradio interface for testing the fine-tuned model.

## Overview

This project uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, allowing you to adapt the large language model with minimal computational overhead. The workflow includes:

1. **Model Setup**: Loading Llama-3.2-3B-Instruct with 4-bit quantization
2. **LoRA Configuration**: Setting up trainable adapters on key projection layers
3. **Data Preparation**: Loading and preprocessing the FineTome-100k dataset
4. **Training**: Fine-tuning the model with optimized training parameters
5. **Inference**: Loading the fine-tuned model and generating responses
6. **Interface**: Interactive Gradio chatbot for testing the model

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- At least 8GB VRAM (more recommended for optimal performance)

## Installation

The notebook automatically installs required packages via pip:

```bash
pip install unsloth transformers trl datasets gradio torch wandb
```

### Key Libraries

- **unsloth**: Efficient fine-tuning framework
- **transformers**: Hugging Face transformers library
- **trl**: Trainer library for supervised fine-tuning
- **datasets**: Hugging Face datasets library
- **gradio**: UI framework for model inference
- **torch**: PyTorch deep learning framework

## Workflow

### 1. Model Loading
Loads the Llama-3.2-3B-Instruct model with 4-bit quantization for memory efficiency:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
```

### 2. LoRA Setup
Configures LoRA adapters on specific projection layers to make training efficient:
```python
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
```

### 3. Dataset Processing
Loads the FineTome-100k dataset and converts it to chat format:
- Standardizes the dataset structure
- Applies Llama-3.1 chat template
- Prepares text field for training

### 4. Model Training
Fine-tunes the model using the SFTTrainer with the following parameters:
- **Batch Size**: 2 (per device)
- **Gradient Accumulation Steps**: 4
- **Warmup Steps**: 100
- **Max Steps**: 50
- **Learning Rate**: 2e-4
- **Optimizer**: Paged AdamW 8-bit
- **Precision**: FP16
- **Sequence Length**: 2048 tokens

### 5. Model Saving & Inference
Saves the fine-tuned model and loads it for inference with chat prompt formatting.

### 6. Gradio Interface
Launches an interactive web interface where users can input prompts and receive responses from the fine-tuned model.

## Configuration Parameters

Modify these parameters in the training cell to customize the fine-tuning:

- `per_device_train_batch_size`: Batch size per GPU
- `gradient_accumulation_steps`: Number of steps to accumulate gradients
- `max_steps`: Total training steps
- `learning_rate`: Learning rate for optimization
- `max_seq_length`: Maximum sequence length
- `r`: LoRA rank (lower = smaller adapters, faster training)

## Output

- **Trained Model**: Saved to `./finetuned_model/` directory
- **Training Logs**: Saved to `./output/` directory
- **Gradio Interface**: Accessible via web browser (URL provided in notebook output)

## Performance Considerations

- **Memory Usage**: Reduced by unsloth and 4-bit quantization
- **Training Speed**: Optimized with gradient accumulation and LoRA adapters
- **Inference Speed**: Fast inference with 4-bit quantization
- **Model Quality**: Fine-tuned on high-quality FineTome-100k dataset

## Next Steps

1. Adjust training parameters based on your GPU resources
2. Modify the dataset by changing the `load_dataset` source
3. Customize the Gradio interface title and description
4. Deploy the model to production using model serving frameworks

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [FineTome Dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k)

## Notes

- Ensure you have sufficient disk space for model weights
- Consider using Weights & Biases (wandb) for detailed training monitoring
- Adjust `max_steps` and `logging_steps` for your use case
- GPU selection may affect training time significantly
