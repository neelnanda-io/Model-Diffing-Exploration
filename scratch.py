# %%
from neel.imports import *
import transformer_lens
from transformer_lens import (
    HookedTransformerConfig,
    HookedTransformer,
    FactoredMatrix,
    ActivationCache,
)
import sae_lens
from sae_lens import HookedSAETransformer
from neel_plotly import *

# %%
torch.set_grad_enabled(False)
# %%
base_model = HookedSAETransformer.from_pretrained("gemma-2-2b")
tuned_model = HookedSAETransformer.from_pretrained("gemma-2-2b-it")
d_model = base_model.cfg.d_model
d_head = base_model.cfg.d_head
n_layers = base_model.cfg.n_layers
print(base_model.cfg)

model = base_model
chat_model = tuned_model

# %%
import torch
from torch import nn
import pprint
import torch.nn.functional as F
from typing import Optional, Union
from huggingface_hub import hf_hub_download, notebook_login
import json
import einops
import plotly.express as px

from typing import NamedTuple
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


class LossOutput(NamedTuple):
    # loss: torch.Tensor
    l2_loss: torch.Tensor
    l1_loss: torch.Tensor
    l0_loss: torch.Tensor
    explained_variance: torch.Tensor
    explained_variance_A: torch.Tensor
    explained_variance_B: torch.Tensor


class CrossCoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = self.cfg["dict_size"]
        d_in = self.cfg["d_in"]
        self.dtype = DTYPES[self.cfg["enc_dtype"]]
        torch.manual_seed(self.cfg["seed"])
        # hardcoding n_models to 2
        self.W_enc = nn.Parameter(torch.empty(2, d_in, d_hidden, dtype=self.dtype))
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(torch.empty(d_hidden, 2, d_in, dtype=self.dtype))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.normal_(torch.empty(d_hidden, 2, d_in, dtype=self.dtype))
        )
        # Make norm of W_dec 0.1 for each column, separate per layer
        self.W_dec.data = (
            self.W_dec.data
            / self.W_dec.data.norm(dim=-1, keepdim=True)
            * self.cfg["dec_init_norm"]
        )
        # Initialise W_enc to be the transpose of W_dec
        self.W_enc.data = einops.rearrange(
            self.W_dec.data.clone(),
            "d_hidden n_models d_model -> n_models d_model d_hidden",
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros((2, d_in), dtype=self.dtype))
        self.d_hidden = d_hidden

        self.to(self.cfg["device"])
        self.save_dir = None
        self.save_version = 0

    def encode(self, x, apply_relu=True):
        # x: [batch, n_models, d_model]
        x_enc = einops.einsum(
            x,
            self.W_enc,
            "batch n_models d_model, n_models d_model d_hidden -> batch d_hidden",
        )
        if apply_relu:
            acts = F.relu(x_enc + self.b_enc)
        else:
            acts = x_enc + self.b_enc
        return acts

    def decode(self, acts):
        # acts: [batch, d_hidden]
        acts_dec = einops.einsum(
            acts,
            self.W_dec,
            "batch d_hidden, d_hidden n_models d_model -> batch n_models d_model",
        )
        return acts_dec + self.b_dec

    def forward(self, x):
        # x: [batch, n_models, d_model]
        acts = self.encode(x)
        return self.decode(acts)

    def get_losses(self, x):
        # x: [batch, n_models, d_model]
        x = x.to(self.dtype)
        acts = self.encode(x)
        # acts: [batch, d_hidden]
        x_reconstruct = self.decode(acts)
        diff = x_reconstruct.float() - x.float()
        squared_diff = diff.pow(2)
        l2_per_batch = einops.reduce(
            squared_diff, "batch n_models d_model -> batch", "sum"
        )
        l2_loss = l2_per_batch.mean()

        total_variance = einops.reduce(
            (x - x.mean(0)).pow(2), "batch n_models d_model -> batch", "sum"
        )
        explained_variance = 1 - l2_per_batch / total_variance

        per_token_l2_loss_A = (
            (x_reconstruct[:, 0, :] - x[:, 0, :]).pow(2).sum(dim=-1).squeeze()
        )
        total_variance_A = (x[:, 0, :] - x[:, 0, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_A = 1 - per_token_l2_loss_A / total_variance_A

        per_token_l2_loss_B = (
            (x_reconstruct[:, 1, :] - x[:, 1, :]).pow(2).sum(dim=-1).squeeze()
        )
        total_variance_B = (x[:, 1, :] - x[:, 1, :].mean(0)).pow(2).sum(-1).squeeze()
        explained_variance_B = 1 - per_token_l2_loss_B / total_variance_B

        decoder_norms = self.W_dec.norm(dim=-1)
        # decoder_norms: [d_hidden, n_models]
        total_decoder_norm = einops.reduce(
            decoder_norms, "d_hidden n_models -> d_hidden", "sum"
        )
        l1_loss = (acts * total_decoder_norm[None, :]).sum(-1).mean(0)

        l0_loss = (acts > 0).float().sum(-1).mean()

        return LossOutput(
            l2_loss=l2_loss,
            l1_loss=l1_loss,
            l0_loss=l0_loss,
            explained_variance=explained_variance,
            explained_variance_A=explained_variance_A,
            explained_variance_B=explained_variance_B,
        )

    @classmethod
    def load_from_hf(
        cls,
        repo_id: str = "ckkissane/crosscoder-gemma-2-2b-model-diff",
        path: str = "blocks.14.hook_resid_pre",
        device: Optional[Union[str, torch.device]] = None,
    ) -> "CrossCoder":
        """
        Load CrossCoder weights and config from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID
            path: Path within the repo to the weights/config
            model: The transformer model instance needed for initialization
            device: Device to load the model to (defaults to cfg device if not specified)

        Returns:
            Initialized CrossCoder instance
        """

        # Download config and weights
        config_path = hf_hub_download(repo_id=repo_id, filename=f"{path}/cfg.json")
        weights_path = hf_hub_download(
            repo_id=repo_id, filename=f"{path}/cc_weights.pt"
        )

        # Load config
        with open(config_path, "r") as f:
            cfg = json.load(f)

        # Override device if specified
        if device is not None:
            cfg["device"] = str(device)

        # Initialize CrossCoder with config
        instance = cls(cfg)

        # Load weights
        state_dict = torch.load(weights_path, map_location=cfg["device"])
        instance.load_state_dict(state_dict)

        return instance
# %%
cross_coder = CrossCoder.load_from_hf()
cross_coder
# %%
norms = cross_coder.W_dec.norm(dim=-1)
norms.shape
relative_norms = norms[:, 1] / norms.sum(dim=-1)
relative_norms.shape
fig = px.histogram(
    relative_norms.detach().cpu().numpy(),
    title="Gemma 2 2B Base vs IT Model Diff",
    labels={"value": "Relative decoder norm strength"},
    nbins=200,
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents")

# Update x-axis ticks
fig.update_xaxes(
    tickvals=[0, 0.25, 0.5, 0.75, 1.0], ticktext=["0", "0.25", "0.5", "0.75", "1.0"]
)

fig.show()
shared_latent_mask = (relative_norms < 0.7) & (relative_norms > 0.3)
shared_latent_mask.shape
cosine_sims = (cross_coder.W_dec[:, 0, :] * cross_coder.W_dec[:, 1, :]).sum(dim=-1) / (
    cross_coder.W_dec[:, 0, :].norm(dim=-1) * cross_coder.W_dec[:, 1, :].norm(dim=-1)
)
cosine_sims.shape
fig = px.histogram(
    cosine_sims[shared_latent_mask].to(torch.float32).detach().cpu().numpy(),
    # title="Cosine similarity of decoder vectors between models",
    log_y=True,  # Sets the y-axis to log scale
    range_x=[-1, 1],  # Sets the x-axis range from -1 to 1
    nbins=100,  # Adjust this value to change the number of bins
    labels={"value": "Cosine similarity of decoder vectors between models"},
)

fig.update_layout(showlegend=False)
fig.update_yaxes(title_text="Number of Latents (log scale)")

fig.show()
# %%
from datasets import load_dataset


def load_pile_lmsys_mixed_tokens():
    try:
        print("Loading data from disk")
        all_tokens = torch.load(
            "/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt"
        )
    except:
        print("Data is not cached. Loading data from HF")
        data = load_dataset(
            "ckkissane/pile-lmsys-mix-1m-tokenized-gemma-2",
            split="train",
            cache_dir="/workspace/cache/",
        )
        data.save_to_disk("/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.hf")
        data.set_format(type="torch", columns=["input_ids"])
        all_tokens = data["input_ids"]
        torch.save(all_tokens, "/workspace/data/pile-lmsys-mix-1m-tokenized-gemma-2.pt")
        print(f"Saved tokens to disk")
    return all_tokens


all_tokens = load_pile_lmsys_mixed_tokens()

# %%
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("lmsys/lmsys-chat-1m")
# %%
lm_sys_ds = ds["train"]
# %%
len(lm_sys_ds)
histogram([len(x) for x in lm_sys_ds])
# %%
import copy

# cross_coder = copy.deepcopy(cross_coder)


def fold_activation_scaling_factor(
    cross_coder, base_scaling_factor, chat_scaling_factor
):
    cross_coder.W_enc.data[0, :, :] = (
        cross_coder.W_enc.data[0, :, :] * base_scaling_factor
    )
    cross_coder.W_enc.data[1, :, :] = (
        cross_coder.W_enc.data[1, :, :] * chat_scaling_factor
    )

    cross_coder.W_dec.data[:, 0, :] = (
        cross_coder.W_dec.data[:, 0, :] / base_scaling_factor
    )
    cross_coder.W_dec.data[:, 1, :] = (
        cross_coder.W_dec.data[:, 1, :] / chat_scaling_factor
    )

    cross_coder.b_dec.data[0, :] = cross_coder.b_dec.data[0, :] / base_scaling_factor
    cross_coder.b_dec.data[1, :] = cross_coder.b_dec.data[1, :] / chat_scaling_factor
    return cross_coder


base_estimated_scaling_factor = 0.2758961493232058
cat_estimated_scaling_factor = 0.24422852496546169
cross_coder = fold_activation_scaling_factor(
    cross_coder, base_estimated_scaling_factor, cat_estimated_scaling_factor
)
# cross_coder = cross_coder.to(torch.bfloat16)

# %%
from functools import partial


def splice_act_hook(act, hook, spliced_act):
    act[:, 1:, :] = spliced_act  # Drop BOS
    return act


def zero_ablation_hook(act, hook):
    act[:] = 0
    return act


def get_ce_recovered_metrics(tokens, model_A, model_B, cross_coder):
    # get clean loss
    ce_clean_A = model_A(tokens, return_type="loss")
    ce_clean_B = model_B(tokens, return_type="loss")

    # get zero abl loss
    ce_zero_abl_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )
    ce_zero_abl_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(cross_coder.cfg["hook_point"], zero_ablation_hook)],
    )

    # bunch of annoying set up for splicing
    _, cache_A = model_A.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_A = cache_A[cross_coder.cfg["hook_point"]]

    _, cache_B = model_B.run_with_cache(
        tokens,
        names_filter=cross_coder.cfg["hook_point"],
        return_type=None,
    )
    resid_act_B = cache_B[cross_coder.cfg["hook_point"]]

    cross_coder_input = torch.stack([resid_act_A, resid_act_B], dim=0)
    cross_coder_input = cross_coder_input[:, :, 1:, :]  # Drop BOS
    cross_coder_input = einops.rearrange(
        cross_coder_input,
        "n_models batch seq_len d_model -> (batch seq_len) n_models d_model",
    )

    cross_coder_output = cross_coder.decode(cross_coder.encode(cross_coder_input))
    cross_coder_output = einops.rearrange(
        cross_coder_output,
        "(batch seq_len) n_models d_model -> n_models batch seq_len d_model",
        batch=tokens.shape[0],
    )
    cross_coder_output_A = cross_coder_output[0]
    cross_coder_output_B = cross_coder_output[1]

    # get spliced loss
    ce_loss_spliced_A = model_A.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (
                cross_coder.cfg["hook_point"],
                partial(splice_act_hook, spliced_act=cross_coder_output_A),
            )
        ],
    )
    ce_loss_spliced_B = model_B.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[
            (
                cross_coder.cfg["hook_point"],
                partial(splice_act_hook, spliced_act=cross_coder_output_B),
            )
        ],
    )

    # compute % CE recovered metric
    ce_recovered_A = 1 - (
        (ce_loss_spliced_A - ce_clean_A) / (ce_zero_abl_A - ce_clean_A)
    )
    ce_recovered_B = 1 - (
        (ce_loss_spliced_B - ce_clean_B) / (ce_zero_abl_B - ce_clean_B)
    )

    metrics = {
        "ce_loss_spliced_A": ce_loss_spliced_A.item(),
        "ce_loss_spliced_B": ce_loss_spliced_B.item(),
        "ce_clean_A": ce_clean_A.item(),
        "ce_clean_B": ce_clean_B.item(),
        "ce_zero_abl_A": ce_zero_abl_A.item(),
        "ce_zero_abl_B": ce_zero_abl_B.item(),
        "ce_diff_A": (ce_loss_spliced_A - ce_clean_A).item(),
        "ce_diff_B": (ce_loss_spliced_B - ce_clean_B).item(),
        "ce_recovered_A": ce_recovered_A.item(),
        "ce_recovered_B": ce_recovered_B.item(),
    }
    return metrics


tokens = all_tokens[torch.randperm(len(all_tokens))[:1]]
ce_metrics = get_ce_recovered_metrics(
    tokens, base_model, tuned_model, cross_coder
)

# %%
# !pip install git+https://github.com/ckkissane/sae_vis.git@crosscoder-vis
from sae_vis.model_fns import CrossCoderConfig, CrossCoder

# encoder_cfg = CrossCoderConfig(d_in=base_model.cfg.d_model, d_hidden=cross_coder.cfg["dict_size"], apply_b_dec_to_input=False)
# sae_vis_cross_coder = CrossCoder(encoder_cfg)
# sae_vis_cross_coder.load_state_dict(folded_cross_coder.state_dict())
# sae_vis_cross_coder = sae_vis_cross_coder.to("cuda:0")
# sae_vis_cross_coder = sae_vis_cross_coder.to(torch.bfloat16)

from sae_vis.data_config_classes import SaeVisConfig
test_feature_idx = [2325,12698,15]
sae_vis_config = SaeVisConfig(
    hook_point = cross_coder.cfg["hook_point"],
    features = test_feature_idx,
    verbose = True,
    minibatch_size_tokens=4,
    minibatch_size_features=16,
)

from sae_vis.data_storing_fns import SaeVisData
sae_vis_data = SaeVisData.create(
    encoder = cross_coder,
    encoder_B = None,
    model_A = base_model,
    model_B = tuned_model,
    tokens = all_tokens[:128], # in practice, better to use more data
    cfg = sae_vis_config,
)

import os
import http
import socketserver
import threading
# from google.colab import output
from IPython.display import IFrame

PORT = 8000

def display_vis_inline(filename: str, height: int = 850):
    '''
    Displays the HTML files in Jupyter notebook. Uses global `PORT` variable defined in prev cell,
    so that each vis has a unique port without having to define a port within the function.
    '''
    global PORT
    from IPython.display import IFrame
    import os
    import http.server
    import socketserver
    import threading

    def serve(directory):
        original_dir = os.getcwd()  # Save current directory
        os.chdir(directory)  # Change to target directory

        # Create a handler for serving files
        handler = http.server.SimpleHTTPRequestHandler

        # Create a socket server with the handler
        try:
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"Serving files from {directory} on port {PORT}")
                httpd.serve_forever()
        except OSError as e:
            print(f"Port {PORT} is already in use. Try a different port.")
            os.chdir(original_dir)  # Restore original directory
            return

    # Start server in the current directory instead of /content
    current_dir = os.getcwd()
    thread = threading.Thread(target=serve, args=(current_dir,), daemon=True)  # Added daemon=True
    thread.start()

    return IFrame(src=f"http://localhost:{PORT}/{filename}", width="100%", height=height)

    PORT += 1

filename = "_feature_vis_demo.html"
sae_vis_data.save_feature_centric_vis(filename)

display_vis_inline(filename)
# %%
# tokenizer = base_model.tokenizer
FAKE_USER_PREFIX = torch.tensor([2, 2224, 235292, 108], device="cuda")
FAKE_CHAT_PREFIX = torch.tensor([108, 51190, 235292, 108], device="cuda")

CTRL_USER_PREFIX = torch.tensor([2, 106, 108], device="cuda")
CTRL_CHAT_PREFIX = torch.tensor([107, 108, 106, 2516, 108], device="cuda")


def make_prompt(user_prompt, assistant_prompt=None):
    user_tokens = model.to_tokens(user_prompt).squeeze(0)
    if assistant_prompt is None:
        return torch.cat([CTRL_USER_PREFIX, user_tokens[1:], CTRL_CHAT_PREFIX], dim=0)[
            None
        ]
    else:
        return torch.cat(
            [
                CTRL_USER_PREFIX,
                user_tokens[1:],
                CTRL_CHAT_PREFIX,
                model.to_tokens(assistant_prompt).squeeze(0)[1:],
            ],
            dim=0,
        )[None]


class Convo:
    def __init__(self, user_prompt: str, assistant_response: str):
        """Initialize a conversation with user and assistant messages"""
        self.user_prompt = user_prompt
        self.assistant_response = assistant_response
        # self.tokenizer = tokenizer

        # Store tokenized versions
        self.user_tokens = model.to_tokens(user_prompt, prepend_bos=False).squeeze(0)
        self.assistant_tokens = model.to_tokens(assistant_response, prepend_bos=False).squeeze(0)

    def get_gemma_format(self):
        """Returns tokens in Gemma chat format with special tokens"""

        return torch.cat([
            CTRL_USER_PREFIX,
            self.user_tokens,
            CTRL_CHAT_PREFIX,
            self.assistant_tokens
        ])

    def get_plain_format(self):
        """Returns tokens with 'User:' and 'Assistant:' prefixes"""
        return torch.cat([
            FAKE_USER_PREFIX,
            self.user_tokens,
            FAKE_CHAT_PREFIX,
            self.assistant_tokens
        ])
    
    def tokens(self, control_prefix=False):
        if control_prefix:
            return self.get_gemma_format()
        else:
            return self.get_plain_format()

    def get_assistant_slice(self, control_prefix=False):
        """Returns the slice indices for extracting assistant response from full sequence"""
        # Find where assistant tokens begin in the full sequence
        if control_prefix:
            start_idx = len(CTRL_USER_PREFIX) + len(self.user_tokens) + len(CTRL_CHAT_PREFIX)
        else:
            start_idx = len(FAKE_USER_PREFIX) + len(self.user_tokens) + len(FAKE_CHAT_PREFIX)
        return slice(start_idx, start_idx + len(self.assistant_tokens))

    def __str__(self):
        return f"User:\n{self.user_prompt}\nAssistant:\n{self.assistant_response}"
    
    def prompt_str(self):
        return f"User:\n{self.user_prompt}\nAssistant:\n"
    
    def gemma_prompt_str(self):
        return f"{model.to_string(CTRL_USER_PREFIX[1:])}{self.user_prompt}{model.to_string(CTRL_CHAT_PREFIX)}"

convo = Convo("Hello, how are you?", "I'm doing great, thank you!")
# %%
t = convo.get_gemma_format()
print(t)
print(t.shape)
print(model.to_string(t))
print(t[convo.get_assistant_slice()])
print(model.to_string(t[convo.get_assistant_slice()]))

print()
print()
t = convo.get_plain_format()
print(t)
print(t.shape)
print(model.to_string(t))
print(t[convo.get_assistant_slice()])
print(model.to_string(t[convo.get_assistant_slice()]))
# %%
tokens = convo.tokens(True)
chat_logits = chat_model(tokens)[0, convo.get_assistant_slice(), :]
chat_log_probs = chat_logits.log_softmax(dim=-1)
chat_gemma_clps = (chat_log_probs[np.arange(len(chat_log_probs)-1), convo.assistant_tokens[1:]])

tokens = convo.tokens(False)
chat_logits = chat_model(tokens)[0, convo.get_assistant_slice(), :]
chat_log_probs = chat_logits.log_softmax(dim=-1)
chat_fake_clps = (chat_log_probs[np.arange(len(chat_log_probs)-1), convo.assistant_tokens[1:]])

tokens = convo.tokens(True)
base_logits = base_model(tokens)[0, convo.get_assistant_slice(), :]
base_log_probs = base_logits.log_softmax(dim=-1)
base_gemma_clps = (base_log_probs[np.arange(len(base_log_probs)-1), convo.assistant_tokens[1:]])

tokens = convo.tokens(False)
base_logits = base_model(tokens)[0, convo.get_assistant_slice(), :]
base_log_probs = base_logits.log_softmax(dim=-1)
base_fake_clps = (base_log_probs[np.arange(len(base_log_probs)-1), convo.assistant_tokens[1:]])

x = nutils.process_tokens_index(convo.assistant_tokens)[1:]
line([chat_gemma_clps, chat_fake_clps, base_gemma_clps, base_fake_clps], x=x, line_labels=["Chat (Gemma)", "Chat (Fake)", "Base (Gemma)", "Base (Fake)"])
# %%
def display_logits(logits, pos=-1):
    nutils.show_df(nutils.create_vocab_df(logits[0, pos], True).head(10))
# %%
c = Convo("What is your name?", "My name is")
prompt = str(c)
# prompt = make_prompt("What is your name?", "My name is")
base_logits, base_cache = base_model.run_with_cache((prompt))
chat_logits, chat_cache = chat_model.run_with_cache((prompt))
print(f"Base")
display_logits(base_logits)
print(f"Chat")
display_logits(chat_logits)
# %%
ONE_TOKEN = model.to_tokens("1", prepend_bos=False).item()
TWO_TOKEN = model.to_tokens("2", prepend_bos=False).item()

base_logit_diff = base_logits[0, -1, TWO_TOKEN] - base_logits[0, -1, ONE_TOKEN]
chat_logit_diff = chat_logits[0, -1, TWO_TOKEN] - chat_logits[0, -1, ONE_TOKEN]
print(f"{base_logit_diff=}\n{chat_logit_diff=}")
print(f"{base_logits[0, -1, TWO_TOKEN]=}\n{base_logits[0, -1, ONE_TOKEN]=}")
print(f"{chat_logits[0, -1, TWO_TOKEN]=}\n{chat_logits[0, -1, ONE_TOKEN]=}")


# base_W_U_fold = base_model.W_U * base_model.ln_final.w[:, None]
# chat_W_U_fold = chat_model.W_U * chat_model.ln_final.w[:, None]
base_logit_diff_dir = (
    (base_model.W_U[:, TWO_TOKEN]
    - base_model.W_U[:, ONE_TOKEN]) / base_cache["scale"][0, -1]
)
chat_logit_diff_dir = (
    chat_model.W_U[:, TWO_TOKEN]
    - chat_model.W_U[:, ONE_TOKEN]
) / chat_cache["scale"][0, -1]
print(f"{base_logit_diff_dir @ base_cache['resid_post', -1][0, -1]=}")
print(f"{chat_logit_diff_dir @ chat_cache['resid_post', -1][0, -1]=}")


base_resids, base_labels = base_cache.decompose_resid(apply_ln=False, pos_slice=-1, return_labels=True)
chat_resids, chat_labels = chat_cache.decompose_resid(apply_ln=False, pos_slice=-1, return_labels=True)
# %%
GEMMA_TOKEN = model.to_tokens(" Gemma", prepend_bos=False).item()
# TWO_TOKEN = model.to_tokens("2", prepend_bos=False).item()

# base_logit_diff = base_logits[0, -1, TWO_TOKEN] - base_logits[0, -1, GEMMA_TOKEN]
# chat_logit_diff = chat_logits[0, -1, TWO_TOKEN] - chat_logits[0, -1, GEMMA_TOKEN]
# print(f"{base_logit_diff=}\n{chat_logit_diff=}")
# print(f"{base_logits[0, -1, TWO_TOKEN]=}\n{base_logits[0, -1, GEMMA_TOKEN]=}")
# print(f"{chat_logits[0, -1, TWO_TOKEN]=}\n{chat_logits[0, -1, GEMMA_TOKEN]=}")


# base_W_U_fold = base_model.W_U * base_model.ln_final.w[:, None]
# chat_W_U_fold = chat_model.W_U * chat_model.ln_final.w[:, None]
base_logit_diff_dir = (
    base_model.W_U[:, GEMMA_TOKEN]
) / base_cache["scale"][0, -1]
chat_logit_diff_dir = (
    chat_model.W_U[:, GEMMA_TOKEN]
) / chat_cache["scale"][0, -1]
print(f"{base_logit_diff_dir @ base_cache['resid_post', -1][0, -1]=}")
print(f"{chat_logit_diff_dir @ chat_cache['resid_post', -1][0, -1]=}")


base_resids, base_labels = base_cache.decompose_resid(
    apply_ln=False, pos_slice=-1, return_labels=True
)
chat_resids, chat_labels = chat_cache.decompose_resid(
    apply_ln=False, pos_slice=-1, return_labels=True
)
print(f"{base_resids.shape=}")
# %%
line([
    base_resids.squeeze(1) @ base_logit_diff_dir,
    chat_resids.squeeze(1) @ chat_logit_diff_dir,
    -(base_resids.squeeze(1) @ base_logit_diff_dir
    - chat_resids.squeeze(1) @ chat_logit_diff_dir),
], x=base_labels, title="DLA per layer to Gemma", line_labels=["Base", "Chat", "Diff"])
line([base_resids.squeeze(1) @ model.W_U.mean(-1), chat_resids.squeeze(1) @ model.W_U.mean(-1)], x=base_labels, title="DLA per layer to W_U.mean(-1)")
line(
    [
        base_resids.squeeze(1) @ base_logit_diff_dir
        - base_resids.squeeze(1) / base_cache["scale"][0, -1] @ model.W_U.mean(-1),
        chat_resids.squeeze(1) @ chat_logit_diff_dir
        - chat_resids.squeeze(1) / chat_cache["scale"][0, -1] @ model.W_U.mean(-1),
        -(
            base_resids.squeeze(1) @ base_logit_diff_dir
            - base_resids.squeeze(1) / base_cache["scale"][0, -1] @ model.W_U.mean(-1)
            - chat_resids.squeeze(1) @ chat_logit_diff_dir
            + chat_resids.squeeze(1) / chat_cache["scale"][0, -1] @ model.W_U.mean(-1)
        ),
    ],
    x=base_labels,
    title="DLA per layer to Gemma",
    line_labels=["Base", "Chat", "Diff"],
)
# %%
a = torch.linspace(-100, 100, int(1e4))
b = 30 * F.tanh(a / 30)
# line(y=b, x=a, include_diag=True)
def invert_monotonic_fn(fn, y_value, x_min=-100, x_max=100, n_points=10000):
    """
    Inverts a monotonic function by using linear interpolation.

    Args:
        fn: The function to invert
        y_value: The output value we want to find the input for
        x_min: Minimum x value to consider
        x_max: Maximum x value to consider
        n_points: Number of points to use for interpolation

    Returns:
        The approximate input value that would produce y_value
    """
    # Create evenly spaced x values
    x_values = torch.linspace(x_min, x_max, n_points)

    # Compute corresponding y values
    y_values = fn(x_values)

    # Find the two points that bracket our target y_value
    if y_values[0] < y_values[-1]:  # Increasing function
        next_idx = torch.searchsorted(y_values, y_value)
    else:  # Decreasing function
        next_idx = torch.searchsorted(-y_values, -y_value)

    prev_idx = next_idx - 1

    # Handle edge cases
    if next_idx == 0:
        return x_values[0]
    if next_idx == n_points:
        return x_values[-1]

    # Linear interpolation between the two nearest points
    x0, x1 = x_values[prev_idx], x_values[next_idx]
    y0, y1 = y_values[prev_idx], y_values[next_idx]

    # Return linearly interpolated value
    return x0 + (x1 - x0) * (y_value - y0) / (y1 - y0)
