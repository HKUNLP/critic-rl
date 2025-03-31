import argparse

from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32_model = load_state_dict_from_zero_checkpoint(model, args.checkpoint_dir)
    fp32_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
