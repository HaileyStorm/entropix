import gradio as gr
import torch
from entropix.config import LLAMA_1B_PARAMS
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_weights import XfmrWeights, LayerWeights, load_weights
from entropix.torch_sampler import sample, calculate_metrics
from entropix.prompts import prompt, bp1

METRIC_RANGES = {
    "logits_entropy": (0.0, 10.0),
    "logits_varentropy": (0.0, 18.0),
    "attn_entropy": (11.1, 12.0),
    "attn_varentropy": (0.0, 2.0),  # This is always zero for me, so, no idea on a decent range
    "agreement": (8.9e-07, 4.75e-05),
    "interaction_strength": (0.05, 3.25),
    "token_probability": (1.0e-12, 0.001),
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

# Preload model and tokenizer
model_params = LLAMA_1B_PARAMS
xfmr_weights = load_weights()
tokenizer = Tokenizer('entropix/tokenizer.model')


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask


def normalize_color(color, color_metric):
    min_val, max_val = METRIC_RANGES[color_metric]
    return max(0.0, min((color - min_val) / (max_val - min_val), 1.0))


def round_to_nearest_05(value):
    scaled_value = value * 100
    rounded_scaled_value = round(scaled_value / 5) * 5
    return rounded_scaled_value / 100


def generate(tokens, color_metric):
    cur_pos = 0
    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta,
                                     model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads,
                          model_params.head_dim).to(device)

    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache,
                                 attn_mask=attn_mask)
    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
    gen_tokens = next_token
    cur_pos = seqlen

    stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

    # Yield the first token
    first_token = tokenizer.decode(next_token.tolist()[0])
    yield first_token, 0.0  # Use 0 as the initial color value

    while cur_pos < 8192:
        cur_pos += 1
        logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos - 1,
                                              freqs_cis[cur_pos - 1:cur_pos], kvcache)
        metrics = calculate_metrics(logits, scores)

        if color_metric == "token_probability":
            probs = torch.softmax(logits[:, -1], dim=-1)
            token_id = next_token.tolist()[0][0]
            color = probs[0, token_id].item()
        else:
            color = metrics.get(color_metric, 0.0)

        color = normalize_color(color if isinstance(color, float) else color.item(), color_metric)
        color = round_to_nearest_05(color)

        next_token = sample(gen_tokens, logits, scores)
        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
        decoded_token = tokenizer.decode(next_token.tolist()[0])
        yield decoded_token, color
        if torch.isin(next_token, stop).any():
            break


def process_input(system_prompt, prompt, color_metric):
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    raw_tokens = tokenizer.encode(formatted_prompt, bos=False, eos=False, allowed_special='all')

    output_text = []
    #colors = []  # If tracking colors to test ranges, make sure to comment out normalization and rounding in `generate`.
    for token, color in generate(raw_tokens, color_metric):
    #    colors.append(color)
        output_text.append((token, color))
        yield output_text
    #print(f"min: {min(x for x in colors if x != 0)}, max: {max(colors)}")
    yield output_text


def main():
    color_options = ["logits_entropy", "logits_varentropy", "attn_entropy", "attn_varentropy", "agreement",
                     "interaction_strength", "token_probability"]

    with gr.Blocks() as demo:
        gr.Markdown("# Entropix Visualizer")
        with gr.Row():
            with gr.Column(scale=1):
                system_prompt = gr.Textbox(value="You are a helpful assistant.", label="System Prompt")
                prompt = gr.Textbox(label="Prompt", lines=4)
                color_metric = gr.Dropdown(choices=color_options, value=color_options[0], label="Color Metric")
                send_btn = gr.Button("Send")
            with gr.Column(scale=3):
                output_text = gr.HighlightedText(label="Generated Story", combine_adjacent=True, show_legend=False)

        send_btn.click(process_input, inputs=[system_prompt, prompt, color_metric], outputs=output_text)

    demo.queue().launch()


if __name__ == "__main__":
    main()