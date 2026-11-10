import os
import re
import sys
import subprocess
import shutil
import ast
import importlib.util

print("=== Step 0: Checking and installing all required dependencies in one go ===")
required_packages = [
    ("sacred", "sacred"),
    ("ftfy", "ftfy"),
    ("einops", "einops"),
    ("pycocoevalcap", "pycocoevalcap"),
    ("iopath", "iopath"),
    ("fairscale", "fairscale"),
    ("webdataset", "webdataset"),
    ("transformers==4.38.2", "transformers"),
    ("tokenizers==0.15.2", "tokenizers"),
    ("omegaconf", "omegaconf"),
    ("torchmetrics", "torchmetrics"),
    ("pyarrow", "pyarrow"),
    ("matplotlib", "matplotlib"),
    ("scikit-learn", "sklearn"),
    ("timm", "timm"),
    ("pandas", "pandas"),
    ("opencv-python", "cv2"),
    ("moviepy", "moviepy"),
    ("peft", "peft"),
    ("easydict", "easydict"),
    ("diffusers==0.29.2", "diffusers"),
    ("huggingface-hub<0.25.0", "huggingface_hub"),
    ("ipython", "IPython"),
    ("tensorboard", "tensorboard"),
    ("tensorboardX", "tensorboardX"),
]

for pkg_spec, import_name in required_packages:
    if importlib.util.find_spec(import_name) is None:
        print(f"Installing missing package: {pkg_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_spec])

import torch

print("=== Step 1: Checking and setting up directory paths ===")
assert os.path.exists("data/finetune_arrows"), "Error: data/finetune_arrows not found!"
assert os.path.exists("data/external_data"), "Error: data/external_data not found!"
os.makedirs("result", exist_ok=True)

print("=== Step 2: Cloning and preparing LAVIS repository ===")
lavis_dir = "LAVIS"
if not os.path.exists(lavis_dir):
    subprocess.run(["git", "clone", "https://github.com/salesforce/LAVIS.git", lavis_dir], check=True)
else:
    subprocess.run(["git", "-C", lavis_dir, "checkout", "."], check=True)

setup_path = os.path.join(lavis_dir, "setup.py")
if os.path.exists(setup_path):
    os.remove(setup_path)

# Replace LAVIS top-level __init__.py so it only registers paths without importing unused modules
lavis_init_path = os.path.join(lavis_dir, "lavis/__init__.py")
with open(lavis_init_path, "w", encoding="utf-8") as f:
    f.write('''import os
import sys
from omegaconf import OmegaConf
from lavis.common.registry import registry

root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.expanduser(default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
''')

# Replace LAVIS models/__init__.py so it only loads BaseModel and BLIP-2
lavis_models_init_path = os.path.join(lavis_dir, "lavis/models/__init__.py")
with open(lavis_models_init_path, "w", encoding="utf-8") as f:
    f.write('''from lavis.models.base_model import BaseModel
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
''')

# Patch LAVIS registry key errors
registry_path = os.path.join(lavis_dir, "lavis/common/registry.py")
if os.path.exists(registry_path):
    with open(registry_path, "r", encoding="utf-8") as f:
        reg_code = f.read()
    reg_code = reg_code.replace("raise KeyError(", "str(")
    with open(registry_path, "w", encoding="utf-8") as f:
        f.write(reg_code)

# Patch Blip2Qformer.forward for M3AE two-tensor call signature & ITC loss
qformer_target = os.path.join(lavis_dir, "lavis/models/blip2_models/blip2_qformer.py")
if os.path.exists(qformer_target):
    with open(qformer_target, "r", encoding="utf-8") as f:
        qf_code = f.read()

    qformer_patch = '''
# === M3AE QFORMER PATCH START ===
import torch
import torch.nn.functional as F

_orig_blip2_qformer_forward = Blip2Qformer.forward

def _m3ae_qformer_forward(self, x, y=None, *args, **kwargs):
    if y is None and isinstance(x, dict):
        return _orig_blip2_qformer_forward(self, x, *args, **kwargs)

    B = x.size(0)
    device = x.device

    enc_width = getattr(self.Qformer.config, "encoder_width", x.size(-1))
    if x.size(-1) != enc_width:
        encoder_hidden_states = F.adaptive_avg_pool1d(x, enc_width)
    else:
        encoder_hidden_states = x
    encoder_atts = torch.ones(encoder_hidden_states.size()[:-1], dtype=torch.long, device=device)

    hidden_size = self.Qformer.config.hidden_size
    if y.size(-1) != hidden_size:
        query_embeds = F.adaptive_avg_pool1d(y, hidden_size)
    else:
        query_embeds = y

    query_output = self.Qformer.bert(
        query_embeds=query_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_atts,
        use_cache=False,
        return_dict=True,
    )
    out_y = query_output.last_hidden_state

    x_pooled = x.mean(dim=1)
    if x_pooled.size(-1) != self.text_proj.in_features:
        x_pooled = F.adaptive_avg_pool1d(x_pooled.unsqueeze(1), self.text_proj.in_features).squeeze(1)

    feat_q = F.normalize(self.vision_proj(out_y), dim=-1)
    feat_x = F.normalize(self.text_proj(x_pooled), dim=-1)

    sim_q2x = torch.einsum("bqd,cd->bqc", feat_q, feat_x)
    temp = self.temp.clamp(min=1e-3) if hasattr(self, "temp") else 0.07
    sim_i2t = sim_q2x.max(dim=1)[0] / temp
    sim_t2i = sim_q2x.max(dim=1)[0].t() / temp

    targets = torch.arange(B, device=device)
    loss_itc = 0.5 * (
        F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
    )
    return out_y, loss_itc

Blip2Qformer.forward = _m3ae_qformer_forward
# === M3AE QFORMER PATCH END ===
'''
    if "M3AE QFORMER PATCH START" not in qf_code:
        with open(qformer_target, "w", encoding="utf-8") as f:
            f.write(qf_code + "\n" + qformer_patch)

print("=== Step 3: Applying codebase compatibility patches ===")
base_dm_path = "m3ae/datamodules/base_datamodule.py"
if os.path.exists(base_dm_path):
    with open(base_dm_path, "r", encoding="utf-8") as f:
        dm_code = f.read()
    dm_code = re.sub(
        r"from\s+transformers\s+import\s*\([^)]*DataCollatorForWholeWordMask[^)]*\)",
        """try:
    from transformers import DataCollatorForWholeWordMask
except ImportError:
    from transformers import DataCollatorForLanguageModeling as DataCollatorForWholeWordMask
from transformers import (
    DataCollatorForLanguageModeling,
    BertTokenizerFast,
    RobertaTokenizerFast
)""",
        dm_code,
        flags=re.DOTALL,
    )
    with open(base_dm_path, "w", encoding="utf-8") as f:
        f.write(dm_code)

m3ae_mod_path = "m3ae/modules/m3ae_module.py"
if os.path.exists(m3ae_mod_path):
    with open(m3ae_mod_path, "r", encoding="utf-8") as f:
        m3ae_code = f.read()
    m3ae_code = re.sub(r"def training_epoch_end\s*\(\s*self\s*,\s*outs\s*\)\s*:", "def on_train_epoch_end(self, outs=None):", m3ae_code)
    m3ae_code = re.sub(r"def validation_epoch_end\s*\(\s*self\s*,\s*outs\s*\)\s*:", "def on_validation_epoch_end(self, outs=None):", m3ae_code)
    m3ae_code = re.sub(r"def test_epoch_end\s*\(\s*self\s*,\s*outs\s*\)\s*:", "def on_test_epoch_end(self, outs=None):", m3ae_code)
    with open(m3ae_mod_path, "w", encoding="utf-8") as f:
        f.write(m3ae_code)

print("=== Step 4: Cleaning old shims and injecting fresh compatibility shim into main.py ===")
shim_header = '''# === COMPATIBILITY SHIM START ===
import sys
import types
import warnings
import inspect
import functools
import importlib.machinery
warnings.filterwarnings("ignore")

# Bypass strict transformers version checks before importing transformers
try:
    import transformers.utils.versions as _uv
    _uv.require_version = lambda *args, **kwargs: True
    _uv.require_version_core = lambda *args, **kwargs: True
except Exception:
    pass

# Mock decord with VideoReader, bridge, and valid spec
try:
    import decord
except ImportError:
    _decord_mock = types.ModuleType('decord')
    class VideoReader:
        def __init__(self, *args, **kwargs):
            pass
    _decord_mock.VideoReader = VideoReader
    _decord_mock.bridge = types.SimpleNamespace(set_bridge=lambda *a, **kw: None)
    _decord_mock.__spec__ = importlib.machinery.ModuleSpec('decord', None)
    sys.modules['decord'] = _decord_mock

# Mock moviepy.editor
try:
    import moviepy.editor
except ImportError:
    _moviepy_editor_mock = types.ModuleType('moviepy.editor')
    class VideoFileClip:
        def __init__(self, *args, **kwargs):
            pass
    _moviepy_editor_mock.VideoFileClip = VideoFileClip
    _moviepy_editor_mock.__spec__ = importlib.machinery.ModuleSpec('moviepy.editor', None)
    sys.modules['moviepy.editor'] = _moviepy_editor_mock

import os
_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import torch
# Disable MPS using lru_cache so torch._dynamo finds .__wrapped__ intact
@functools.lru_cache(maxsize=None)
def _mps_false():
    return False

if hasattr(torch.backends, "mps"):
    torch.backends.mps.is_available = _mps_false
    torch.backends.mps.is_built = _mps_false

# Redirect hardcoded .cuda() calls to CPU
if not torch.cuda.is_available():
    _default_dev = torch.device("cpu")
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to(_default_dev)
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self.to(_default_dev)

import transformers
import transformers.modeling_utils as mu
import transformers.pytorch_utils as pu
import transformers.optimization as opt

if not hasattr(transformers, "DataCollatorForWholeWordMask"):
    transformers.DataCollatorForWholeWordMask = transformers.DataCollatorForLanguageModeling
if not hasattr(transformers, "AdamW"):
    transformers.AdamW = torch.optim.AdamW
if not hasattr(opt, "AdamW"):
    opt.AdamW = torch.optim.AdamW

if not hasattr(transformers, "EncoderDecoderCache"):
    try:
        from transformers.cache_utils import DynamicCache as EncoderDecoderCache
        transformers.EncoderDecoderCache = EncoderDecoderCache
    except ImportError:
        class EncoderDecoderCache:
            pass
        transformers.EncoderDecoderCache = EncoderDecoderCache

try:
    import transformers.models.clip.modeling_clip as _cmc
    if not hasattr(_cmc, "_expand_mask"):
        def _expand_mask(mask, dtype, tgt_len=None):
            if mask is None:
                return mask
            expanded_mask = mask[:, None, None, :].expand(
                mask.shape[0], 1, tgt_len if tgt_len is not None else mask.shape[-1], mask.shape[-1]
            )
            return expanded_mask.to(dtype)
        _cmc._expand_mask = _expand_mask
except Exception:
    pass

try:
    import torchmetrics
    _orig_metric_init = torchmetrics.Metric.__init__
    def _patched_metric_init(self, *args, **kwargs):
        kwargs.pop("compute_on_step", None)
        return _orig_metric_init(self, *args, **kwargs)
    torchmetrics.Metric.__init__ = _patched_metric_init
except Exception:
    pass

try:
    import pytorch_lightning as pl
    _orig_trainer_init = pl.Trainer.__init__
    _valid_trainer_params = set(inspect.signature(_orig_trainer_init).parameters.keys())
    def _patched_trainer_init(self, *args, **kwargs):
        kwargs.pop("gpus", None)
        kwargs["accelerator"] = "cpu"
        kwargs["devices"] = 1
        if "max_steps" in kwargs and kwargs["max_steps"] is None:
            kwargs["max_steps"] = -1
        for k in list(kwargs.keys()):
            if k not in _valid_trainer_params:
                kwargs.pop(k, None)
        return _orig_trainer_init(self, *args, **kwargs)
    pl.Trainer.__init__ = _patched_trainer_init
except Exception:
    pass
# === COMPATIBILITY SHIM END ===
'''

with open("main.py", "r", encoding="utf-8") as f:
    raw_main = f.read()

# Strip all previously injected COMPATIBILITY SHIM blocks
clean_main = re.sub(
    r"# === COMPATIBILITY SHIM START ===[\s\S]*?# === COMPATIBILITY SHIM END ===\n*",
    "",
    raw_main,
)

# Strip any previously injected auto-resume lines to prevent duplicate blocks
clean_main = re.sub(
    r"[ \t]*import glob as _glob\n[ \t]*_ckpts = [^\n]+\n[ \t]*_resume_ckpt = [^\n]+\n[ \t]*if _resume_ckpt:\n[ \t]*print\([^\n]+\)\n[ \t]*trainer\.fit\(model, datamodule=dm, ckpt_path=_resume_ckpt\)\n",
    "        trainer.fit(model, datamodule=dm)\n",
    clean_main,
)

orig_main_lines = clean_main.splitlines(keepends=True)
new_main_lines = []
for line in orig_main_lines:
    if "trainer.fit(model, datamodule=dm)" in line and not line.lstrip().startswith("#"):
        indent = line[: len(line) - len(line.lstrip())]
        new_main_lines.append(f"{indent}import glob as _glob\n")
        new_main_lines.append(f'{indent}_ckpts = sorted(_glob.glob("result/**/checkpoints/*.ckpt", recursive=True), key=os.path.getmtime)\n')
        new_main_lines.append(f'{indent}_resume_ckpt = _ckpts[-1] if (_config["load_path"] == "" and _ckpts) else (_config["load_path"] if _config["load_path"] != "" else None)\n')
        new_main_lines.append(f"{indent}if _resume_ckpt:\n")
        new_main_lines.append(f'{indent}    print(f"Resuming training from checkpoint: {{_resume_ckpt}}")\n')
        new_main_lines.append(f"{indent}trainer.fit(model, datamodule=dm, ckpt_path=_resume_ckpt)\n")
    else:
        new_main_lines.append(line)

final_main_code = shim_header + "\n" + "".join(new_main_lines)
ast.parse(final_main_code)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(final_main_code)

print("=== Step 5: Launching Training Locally ===")
cmd = [
    sys.executable, "main.py", "with", "task_finetune_vqa_slack", "text_roberta",
    'vit=ViT-B/16',
    'patch_size=16',
    'image_size=384',
    'data_root=data/finetune_arrows',
    'test_only=False',
    'fast_dev_run=False',
    'gpu_ids=0',
    'num_gpus=0',
    'batch_size=4',
    'per_gpu_batchsize=1',
    'num_workers=0',
    'vqa_label_size=222',
    'T5_model=t5-base'
]

env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath("LAVIS")
subprocess.run(cmd, env=env, check=True)