from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from typing import OrderedDict
import torch
import torch.nn as nn

# 1. 配置类
class E5Config(PretrainedConfig):
    model_type = "e5"
    def __init__(
        self, 
        vocab_size=50257, 
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

# 2. 基础模型类
class E5ForCausalLM(PreTrainedModel):
    config_class = E5Config
    base_model_prefix = "e5"
    
    def __init__(self, config):
        super().__init__(config)
        # 加载预训练模型时使用自定义配置
        self.model = AutoModelForCausalLM.from_pretrained(
            "/data0/pre_trained_model/multilingual_e5_small_51036/",
            trust_remote_code=True,
            config=config  # 传递自定义配置
        )
        # 移除自定义的lm_head，使用模型自带的语言模型头

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 直接使用模型输出的logits
        logits = outputs.logits
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def generate(self, input_ids, max_length=50, num_beams=1, **kwargs):
        # 使用transformers库中的generate方法
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            **kwargs
        )

# 3. 自动模型类
class E5ModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = OrderedDict([
        (E5Config, E5ForCausalLM),
    ])

# 4. 带注意力的模型类
class E5ForCausalLMWithAttn(E5ForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # 确保配置类型正确
        # if isinstance(config, PretrainedConfig):
        #     e5_config = E5Config(
        #         vocab_size=config.vocab_size,
        #         hidden_size=config.hidden_size,
        #         num_hidden_layers=config.num_hidden_layers,
        #         num_attention_heads=config.num_attention_heads,
        #     )
        #     config = e5_config
        
        model = cls(config)
        return model

    def __init__(self, config, attn_implementation=None, **kwargs):
        super().__init__(config)
        if attn_implementation:
            print(f"Warning: E5 does not support attn_implementation={attn_implementation}")

# 5. 注册配置
AutoConfig.register("e5", E5Config)

if __name__ == "__main__":
    path = "/data0/pre_trained_model/multilingual_e5_small_51036/"
    config = AutoConfig.from_pretrained(path)
    model = E5ForCausalLMWithAttn.from_pretrained(
        path, 
        config=config,
        trust_remote_code=True
    )
    print(model.generate)
    print(hasattr(model, 'model'))
    print(hasattr(model, 'transformer'))

    qwen_model = AutoModelForCausalLM.from_pretrained("/data0/pre_trained_model/Qwen2.5-3B-Instruct/",trust_remote_code=True)
    print(qwen_model.generate)
    print(qwen_model)

    embedding_attr = 'model'
    inputs = {
        "input_ids": torch.randint(0, 100, (1, 10)),
        "attention_mask": torch.ones((1, 10))
    }
    outputs = (getattr(qwen_model, embedding_attr))(**inputs)
    print(outputs)
