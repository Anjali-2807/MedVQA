# === COMPATIBILITY SHIM START ===
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

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import copy
import resource
import torch
import pytorch_lightning as pl

from m3ae.config import ex
from m3ae.datamodules.multitask_datamodule import MTDataModule
from m3ae.modules import M3AETransformerSS

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules
    dm = MTDataModule(_config, dist=False)

    # Module
    model = M3AETransformerSS(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    num_gpus = max(num_gpus, 1)
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else -1
    max_epochs = _config["max_epoch"] if max_steps == -1 else 1000

    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = 32
    elif torch.cuda.is_available():
        accelerator = "gpu"
        precision = _config["precision"]
    else:
        accelerator = "cpu"
        precision = 32

    # Trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=1,
        num_nodes=_config["num_nodes"],
        precision=precision,
        benchmark=False,
        deterministic=False,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"]
    )

    if not _config["test_only"]:
        import glob as _glob
        _ckpts = sorted(_glob.glob("result/**/checkpoints/*.ckpt", recursive=True), key=os.path.getmtime)
        _resume_ckpt = _ckpts[-1] if (_config["load_path"] == "" and _ckpts) else (_config["load_path"] if _config["load_path"] != "" else None)
        if _resume_ckpt:
            print(f"Resuming training from checkpoint: {_resume_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=_resume_ckpt)
        if "finetune" in exp_name:
            trainer.test(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)