You need to download the [llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat) model and set your openai api in dall_e3.py before you run this code.

**Quick Start**

```bash
conda env create -f environment.yaml
torchrun --nproc_per_node 1 text_spanning_tree.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 4096 --max_batch_size 6 --word_count 5000 --seed 1
```