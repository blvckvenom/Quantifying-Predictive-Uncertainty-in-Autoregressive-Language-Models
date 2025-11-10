from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .metrics import entropy_from_logits, surprisal_of_true_token

def load_model(model_name="gpt2", device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@torch.no_grad()
def compute_logits(tokenizer, model, device, texts, max_len=128):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=False)
    logits = out.logits.detach().cpu().numpy()  # [batch, steps, vocab]
    return logits, enc["input_ids"].cpu().numpy()
