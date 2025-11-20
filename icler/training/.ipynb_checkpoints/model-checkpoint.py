from dataclasses import dataclass
import logging
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from transformers import (
    AutoModel, 
    BertModel, 
    AutoModelForCausalLM, 
    BertForMaskedLM,
    PreTrainedModel
)
from transformers.file_utils import ModelOutput

from gritlm import GritLM

logger = logging.getLogger(__name__)


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2: return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

class GritLMTrainModel(torch.nn.Module):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        pooling_method: str = "mean",
        emb_loss_fn = None,
        gen_loss_fn = None,
        gen_add_kwargs: dict = None,
        emb_add_kwargs: dict = None,
        model_type: str = "bert"
    ):
        super().__init__()
        
        if isinstance(model, str):
            if model_type == "bert":
                self.model = BertForMaskedLM.from_pretrained(model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model)
        else:
            self.model = model
            
        self.pooling_method = pooling_method
        self.emb_loss_fn = emb_loss_fn
        self.gen_loss_fn = gen_loss_fn
        self.gen_add_kwargs = gen_add_kwargs or {}
        self.emb_add_kwargs = emb_add_kwargs or {}

    def encode(self, features):
        # 对于BERT模型，我们需要获取隐藏状态
        outputs = self.model(
            **features,
            **self.emb_add_kwargs,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 使用最后一层的隐藏状态
        if hasattr(outputs, 'hidden_states'):
            hidden = outputs.hidden_states[-1]
        else:
            hidden = outputs.last_hidden_state
        
        # 实现不同的池化方法
        if self.pooling_method == "mean":
            attention_mask = features.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden.device)
                hidden = hidden * attention_mask.unsqueeze(-1)
                pooled = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
            else:
                pooled = hidden.mean(dim=1)
        elif self.pooling_method == "cls":
            # 使用[CLS]标记的表示
            pooled = hidden[:, 0]
        elif self.pooling_method == "last":
            # 使用最后一个非填充标记的表示
            attention_mask = features.get("attention_mask", None)
            if attention_mask is not None:
                last_indices = attention_mask.sum(dim=1) - 1
                batch_size = hidden.size(0)
                pooled = hidden[torch.arange(batch_size), last_indices]
            else:
                pooled = hidden[:, -1]
        else:
            raise ValueError(
                f"Unsupported pooling method: {self.pooling_method}. "
                "Supported methods are: mean, cls, last"
            )
            
        return pooled

    def forward(self, **kwargs):
        device = next(self.parameters()).device
        for key, value in kwargs.items():
            if isinstance(value, dict):
                kwargs[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in value.items()}
            elif isinstance(value, torch.Tensor):
                kwargs[key] = value.to(device)

        loss_gen = None
        if 'generative' in kwargs and kwargs['generative'] is not None:
            generative = kwargs['generative'].copy()
            
            try:
                outputs = self.model(**generative, **self.gen_add_kwargs)
                loss_gen = outputs.loss
            except Exception as e:
                logger.error(f"Error in generative loss calculation: {str(e)}")
                logger.error(f"Generative inputs: {generative.keys()}")
                raise

        # 处理向量表示任务
        q_reps = kwargs.get('q_reps', None)
        p_reps = kwargs.get('p_reps', None)
        query = kwargs.get('query', None)
        passage = kwargs.get('passage', None)
        q_grad = kwargs.get('q_grad', True)
        p_grad = kwargs.get('p_grad', True)

        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query)

        if (p_reps is None) and (passage is not None):
            if p_grad:
                p_reps = self.encode(passage)
            else:
                with torch.no_grad():
                    p_reps = self.encode(passage)
            
        loss_emb = self.emb_loss_fn(
            q_reps, p_reps
        ) if (q_reps is not None and p_reps is not None) else None        

        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])
        print("loss_emb:",loss_emb)
        print("loss_gen:",loss_gen)

        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)
