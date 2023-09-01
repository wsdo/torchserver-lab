

from abc import ABC
from ts.torch_handler.base_handler import BaseHandler
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

class ChatGLMHandler(BaseHandler, ABC):
    def __init__(self):
        super(ChatGLMHandler, self).__init__()
        self.tokenizer = None
        self.model = None
        self.max_source_length = 1024
        self.max_target_length = 256
        self.logits_processor = LogitsProcessorList()

        class InvalidScoreLogitsProcessor(LogitsProcessor):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if torch.isnan(scores).any() or torch.isinf(scores).any():
                    scores.zero_()
                    scores[..., 5] = 5e4
                return scores

        self.logits_processor.append(InvalidScoreLogitsProcessor())

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_dir,config=config,trust_remote_code=True)
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.model = self.model.to(self.device)

    def preprocess(self, requests):
        # Expecting data in the format: {"context": "some context", "question": "some question"}
        prompts = [f'判例:\n{request["context"]}\n问题:\n{request["question"]}\n答案:\n' for request in requests]
        self.tokenizer.truncation_side = 'left'
        ret = self.tokenizer(
            prompts, 
            max_length=self.max_source_length, 
            truncation=True, 
            padding=True, return_tensors="pt"
        ).to(self.device)
        return ret

    def inference(self, data, *args, **kwargs):
        gen_kwargs = {
            "max_length": self.max_source_length+self.max_target_length, 
            "do_sample": False, 
            "top_p": 0.7,
            "temperature": 0.05, 
            "logits_processor": self.logits_processor,
            **kwargs
        }
        outputs = self.model.generate(**data, **gen_kwargs)
        outputs = outputs.tolist()
        return outputs

    def postprocess(self, inference_output):
        # Formatting the response as per OpenAPI standard
        responses = []
        for prediction in inference_output:
            answer = self.tokenizer.decode(prediction)
            responses.append({"answer": answer})
        return responses

