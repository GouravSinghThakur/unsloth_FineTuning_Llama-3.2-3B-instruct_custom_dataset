# LLaMA 3.2 Fine-Tuning Project

Fine-tune Meta's LLaMA 3.2 3B Instruct model efficiently using Unsloth with 4-bit quantization and LoRA.

## Quick Start

### Install Dependencies
```bash
pip install unsloth transformers trl datasets torch gradio wandb
```

### Run the Notebook
Open `FineTunning_with_llama_unsloth.ipynb` in Jupyter and run all cells.

## What It Does

1. **Load Model** - Downloads and loads LLaMA 3.2 3B with 4-bit quantization
2. **Configure LoRA** - Sets up parameter-efficient fine-tuning
3. **Load Dataset** - Uses FineTome-100k dataset from Hugging Face
4. **Train** - Fine-tunes the model with your data
5. **Save Model** - Exports trained weights
6. **Inference** - Tests the fine-tuned model

## Key Features

- ⚡ **2-5x Faster** - Efficient training with Unsloth
- 💾 **80% Less Memory** - 4-bit quantization + LoRA
- 🎯 **Easy to Use** - Single Jupyter notebook
- 📊 **Track Progress** - Weights & Biases integration

## Files

- `FineTunning_with_llama_unsloth.ipynb` - Main training notebook
- `DATA/` - Training datasets
- `README.md` - Full documentation

## Hardware Requirements

- GPU with 8GB+ VRAM (RTX 3080 recommended)
- CUDA 11.8+

## Datasets in `DATA/`

- `AI_JOB_MARKET.json` - AI job market data
- `clean_dolly_dataset.json` - Cleaned Dolly dataset
- `final_finetuning_dataset.json` - Combined training dataset

## Model Settings

- **Model**: Llama-3.2-3B-Instruct
- **Batch Size**: 2
- **Learning Rate**: 2e-4
- **Max Steps**: 50
- **Sequence Length**: 2048 tokens

## Output

- `finetuned_model/` - Your trained model (generated after training)

## Next Steps

1. Update `max_steps` and learning rate in the notebook
2. Prepare your own datasets
3. Run training and monitor with Weights & Biases
4. Use the inference cell to test your model

---
