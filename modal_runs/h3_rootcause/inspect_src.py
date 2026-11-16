"""Print the library source the root-cause harness has to mirror exactly (run inside the Modal image). Not for merge."""

import inspect

import diffusers
import peft
import torch
from diffusers.models import attention, attention_dispatch
from peft.tuners.lora import layer as lora_layer

print("VERSIONS", torch.__version__, diffusers.__version__, peft.__version__)
for name, obj in [
    ("SwiGLU", attention.SwiGLU if hasattr(attention, "SwiGLU") else None),
    ("FeedForward.forward", attention.FeedForward.forward),
    ("FeedForward.__init__", attention.FeedForward.__init__),
    ("dispatch_attention_fn", attention_dispatch.dispatch_attention_fn),
    ("lora.Linear.forward", lora_layer.Linear.forward),
    ("RMSNorm.forward", torch.nn.RMSNorm.forward),
]:
    print("=" * 20, name)
    if obj is None:
        import diffusers.models.activations as act

        obj = act.SwiGLU
    print(inspect.getsource(obj))
print("=" * 20, "native backend fn")
src = inspect.getsource(attention_dispatch)
i = src.find("def _native_attention")
print(src[i : i + 1800])
print("DEFAULT BACKEND", getattr(attention_dispatch._AttentionBackendRegistry, "_active_backend", None))
