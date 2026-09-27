"""Inference-only compatibility adapter for the pinned official AnyDoor source.

No source files or historical training environments are modified. No generated
placement mask is represented as ground truth. See the pilot review gate.
"""
from __future__ import annotations

import ast
import gc
import hashlib
import importlib
import os
from pathlib import Path
import sys

OFFICIAL_SHA256 = '89be5db3ca8d07dab4d525b7c8d40e12f80c6a45f8fce769d5617640d2f36ec0'
SOURCE_REVISION = '44ca2b2a70ec2cf107f3d26a5b46def6670fb0a5'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def activate(source):
    source = Path(source).resolve()
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / 'dinov2'))
    # Lightning moved this public helper; the helper's behavior is unchanged.
    try:
        importlib.import_module('pytorch_lightning.utilities.distributed')
    except ModuleNotFoundError as exc:
        if exc.name != 'pytorch_lightning.utilities.distributed':
            raise
        sys.modules['pytorch_lightning.utilities.distributed'] = importlib.import_module(
            'pytorch_lightning.utilities.rank_zero')
    return source


def load_pair_functions(source):
    """Load only pure helpers, never official run_inference's import-time GPU load."""
    import cv2
    import numpy as np
    from datasets import data_utils
    path = Path(source) / 'run_inference.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {'process_pairs', 'crop_back'}]
    if len(nodes) != 2:
        raise ValueError('Pinned official helper definitions changed')
    namespace = {**vars(data_utils), 'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['process_pairs'], namespace['crop_back']


def sdpa_forward(self, x, context=None, mask=None):
    import torch.nn.functional as F
    context = x if context is None else context
    b, n, _ = x.shape
    q = self.to_q(x).reshape(b, n, self.heads, -1).transpose(1, 2)
    k = self.to_k(context).reshape(b, -1, self.heads, q.shape[-1]).transpose(1, 2)
    v = self.to_v(context).reshape(b, -1, self.heads, q.shape[-1]).transpose(1, 2)
    attention_mask = None if mask is None else mask.reshape(b, 1, 1, -1).bool()
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask,
                                           dropout_p=0.0, scale=self.scale)
    return self.to_out(result.transpose(1, 2).reshape(b, n, -1))


def verify_attention():
    import torch
    from ldm.modules.attention import CrossAttention
    torch.manual_seed(416)
    module = CrossAttention(32, context_dim=48, heads=4, dim_head=8).eval()
    x, context = torch.randn(2, 17, 32), torch.randn(2, 13, 48)
    with torch.no_grad():
        expected = module(x, context)
        actual = sdpa_forward(module, x, context)
    error = (expected - actual).abs().max().item()
    if not torch.allclose(expected, actual, atol=2e-6, rtol=2e-5):
        raise ValueError(f'SDPA math regression: {error}')
    report = {'cpu_fp32_max_abs_error': error}
    if torch.cuda.is_available():
        module = module.half().cuda()
        x, context = x.half().cuda(), context.half().cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            expected = module(x, context)
            actual = sdpa_forward(module, x, context)
        gpu_error = (expected - actual).abs().max().item()
        if not torch.isfinite(actual).all() or not torch.allclose(expected, actual, atol=2e-3, rtol=2e-2):
            raise ValueError(f'GPU FP16 SDPA regression: {gpu_error}')
        report['gpu_fp16_max_abs_error'] = gpu_error
    return report


def construct_model(source):
    import torch
    from omegaconf import OmegaConf
    source = activate(source)
    from ldm.modules.encoders import modules as encoders
    from ldm.modules.attention import CrossAttention
    from ldm.util import instantiate_from_config

    def dino_init(self, device='cuda', freeze=True):
        torch.nn.Module.__init__(self)
        # Same official architecture, materialized from the full checkpoint
        # below. Avoid a duplicate 4.5GB DINO download/allocation.
        with torch.device('meta'):
            # Both official schedules are all-zero at drop_path_rate=0;
            # uniform avoids Tensor.item() on a meta initialization tensor.
            self.model = encoders.hubconf.dinov2_vitg14(pretrained=False,
                drop_path_rate=0.0, drop_path_uniform=True)
        self.device = device
        self.image_mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 3, 1, 1)
        self.image_std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 3, 1, 1)
        self.projector = torch.nn.Linear(1536, 1024)
        if freeze:
            self.freeze()

    attention_test = verify_attention()
    encoders.FrozenDinoV2Encoder.__init__ = dino_init
    CrossAttention.forward = sdpa_forward
    config = OmegaConf.load(source / 'configs/anydoor.yaml')
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        model = instantiate_from_config(config.model)
    finally:
        torch.set_default_dtype(previous_dtype)
    return model, attention_test


def load_model(source, weights, precision='fp16_weights_autocast_sdpa'):
    import torch
    weights = Path(weights).resolve()
    if sha256(weights) != OFFICIAL_SHA256:
        raise ValueError('Official checkpoint SHA256 mismatch')
    # Official, pinned, hash-verified Lightning checkpoint includes training
    # metadata. mmap prevents allocating all optimizer tensors into RAM.
    checkpoint = torch.load(weights, map_location='cpu', mmap=True, weights_only=False)
    state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if not any(key.startswith('cond_stage_model.model.blocks.') for key in state):
        raise ValueError('Full DINO weights absent; never silently use random DINO')
    model, attention_test = construct_model(source)
    # Keep only shape/dtype, not a second reference to every initialized tensor.
    expected = {name: (tuple(value.shape), value.dtype) for name, value in model.state_dict().items()}
    missing, unexpected = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f'Strict state key mismatch: missing={missing}, unexpected={unexpected}')
    parameter_names = set(dict(model.named_parameters()))
    for name, (shape, dtype) in expected.items():
        tensor = state[name]
        if precision == 'fp32_offload' and tensor.is_floating_point():
            dtype = torch.float32
        if tuple(tensor.shape) != shape:
            raise ValueError(f'Shape mismatch: {name}')
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        # Parameters FP16, diffusion schedule buffers retain their FP32 dtype.
        # Stream each verified tensor directly to GPU. Keeping a second full
        # CPU half-model alongside mmap pages peaked at16GB RSS in the smoke.
        tensor = tensor.to(dtype=dtype, device='cuda', copy=True)
        if name in parameter_names:
            parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
        else:
            parent._buffers[leaf] = tensor
    del expected, checkpoint, state
    gc.collect()
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('Unmaterialized model state')
    model.eval().requires_grad_(False).cuda()
    model.control_scales = [1.0] * 13
    from cldm.ddim_hacked import DDIMSampler
    process_pairs, crop_back = load_pair_functions(source)
    return model, DDIMSampler(model), process_pairs, crop_back, attention_test


def apply_shape_control(item, placement_mask):
    """Use the target silhouette in hint channel4, not its enclosing rectangle.

    Matches the official demo's shape-mask conditioning concept. Keep baseline
    ref/RGB/crop/RNG fixed for a single-variable diagnostic; do not silently adopt
    the demo's different random crop policy. This is conditioning, NEVER GT.
    """
    import cv2
    import numpy as np
    mask = np.asarray(placement_mask)
    if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 1}) or not mask.any():
        raise ValueError('Shape control requires a nonempty binary placement mask')
    y1, y2, x1, x2 = map(int, item['tar_box_yyxx_crop'])
    h1, w1, h2, w2 = map(int, item['extra_sizes'])
    if not (0 <= y1 < y2 <= mask.shape[0] and 0 <= x1 < x2 <= mask.shape[1]):
        raise ValueError('Invalid crop coordinates')
    cropped = mask[y1:y2, x1:x2]
    if cropped.shape != (h1, w1) or h2 < h1 or w2 < w1:
        raise ValueError('Shape control crop/pad geometry mismatch')
    top, left = (h2 - h1) // 2, (w2 - w1) // 2
    padded = np.pad(cropped.astype(np.float32),
        ((top, h2 - h1 - top), (left, w2 - w1 - left)), constant_values=-1)
    hint = item['hint'].copy()
    if hint.ndim != 3 or hint.shape[2] != 4:
        raise ValueError('Expected four-channel AnyDoor conditioning')
    hint[:, :, 3] = cv2.resize(padded, (hint.shape[1], hint.shape[0]), interpolation=cv2.INTER_NEAREST)
    return {**item, 'hint': hint}


def generate(runtime, reference, reference_mask, background, placement_mask, seed, precision='fp16_weights_autocast_sdpa', shape_control=False):
    from contextlib import nullcontext
    import random
    import numpy as np
    import torch
    model, sampler, process_pairs, crop_back, _ = runtime
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    item = process_pairs(reference, reference_mask, background.copy(), placement_mask)
    if shape_control:
        item = apply_shape_control(item, placement_mask)
    if precision == 'fp32_offload':
        model.low_vram_shift(is_diffusing=False)
    control = torch.from_numpy(item['hint'].copy()).permute(2, 0, 1)[None].float().cuda()
    reference_tensor = torch.from_numpy(item['ref'].copy()).permute(2, 0, 1)[None].float().cuda()
    autocast = nullcontext() if precision == 'fp32_offload' else torch.autocast('cuda', dtype=torch.float16)
    with torch.inference_mode(), autocast:
        cond = {'c_concat': [control], 'c_crossattn': [model.get_learned_conditioning(reference_tensor)]}
        uncond = {'c_concat': [control], 'c_crossattn': [model.get_learned_conditioning(torch.zeros_like(reference_tensor))]}
        if precision == 'fp32_offload':
            model.low_vram_shift(is_diffusing=True)
        latent, _ = sampler.sample(50, 1, (4, 64, 64), cond, verbose=False, eta=0.0,
                                   unconditional_guidance_scale=5.0, unconditional_conditioning=uncond)
        if precision == 'fp32_offload':
            model.low_vram_shift(is_diffusing=False)
        decoded = model.decode_first_stage(latent).float()
    if not torch.isfinite(decoded).all():
        raise ValueError('Nonfinite generated pixels')
    pixels = decoded[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5
    # Keep the official 511-row crop for first pilot parity. Disclose this
    # preprocessing behavior in the manifest rather than silently fixing it.
    pixels = np.clip(pixels, 0, 255)[1:, :, :]
    result = crop_back(pixels, background.copy(), item['extra_sizes'], item['tar_box_yyxx_crop'])
    return result.astype(np.uint8), {'crop_yyxx': item['tar_box_yyxx_crop'].tolist(),
                                     'extra_sizes': item['extra_sizes'].tolist(),
                                     'shape_control': shape_control,
                                     'output_mask_verified': False}
