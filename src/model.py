from typing import Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers import LlavaNextForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from collections import defaultdict

CAUSAL_OUTPUTVALUES = ["avg_gen_layer", "avg_ppt_layer", "avg_all_layer", "fst_gen_layer", "last_gen_layer"]


class MMEBModel(nn.Module):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder: PreTrainedModel,
                 tokenizer: AutoTokenizer,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 filter_words: Optional[list[str]] = None,
                 ):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature

        self.filter_ids = None
        if filter_words is not None:
            self.filter_ids  = set()
            for filter_word in filter_words:
                filter_ids |= set(self.tokenizer.encode(filter_word))
            self.filter_ids  = list(filter_ids)

        self.generation_configs = {
            "temperature": temperature,
            "top_p": 0.9,
            "max_new_tokens": 40,
            "do_sample": True
        }
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.generation_configs['pad_token_id'] = self.tokenizer.eos_token_id


    def get_all_output_values(self) -> list[str]:
        # +1 token embedding
        num_layers = self.encoder.config.num_hidden_layers + 1
        all_output_values = []
        for ov in CAUSAL_OUTPUTVALUES:
            if ov.endswith("_"):
                all_output_values.extend([ov+str(l) for l in range(num_layers)])
            else:
                all_output_values.append(ov)
        return all_output_values


    def encode_input(self, input):
        outputs = self.encoder.generate(**input, **self.generation_configs,
                                        return_dict_in_generate=True, output_hidden_states=True, output_scores=True)

        #generations = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
        decoded_ids = outputs['sequences'][:, input['input_ids'].shape[1]:]

        #generations_remove_prompt = self.tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)
        #all_generations += generations

        hidden_states = outputs["hidden_states"]
                
        aggregated = self._aggregate_embeddings(hidden_states, input['attention_mask'], decoded_ids)


        output_embeddings = aggregated[self.pooling]
        
        if self.normalize:
            output_embeddings = torch.nn.functional.normalize(output_embeddings, p=2, dim=1)

        return output_embeddings


    def _aggregate_embeddings(self, hidden_states, attention_mask, decoded_ids):
        """
        aggregate hidden states for embeddings
        """
        embeddings = {}

        # `encoder_hidden_states` is a tuple of layers of hidden states
        # `decoder_hidden_states` is a tuple of tuple. The first tuple means tokens, while the second means layers.
        prompt_hidden_states = hidden_states[0]

        # post-process to remove special tokens in decoded sequences
        # decoding starts from second token
        # the first hidden state is used to generate second token
        # because the first token's hidden state is the last hidden state of prompt
        special_tokens_masks = [self.tokenizer.get_special_tokens_mask(decoded_ids[batch_id][1:], already_has_special_tokens=True) for batch_id in range(len(decoded_ids))]
        # filter out `self.filter_ids`
        if self.filter_ids is not None:
            # the hidden states that are used to predict the next token
            special_tokens_masks = [
                [1 if token_id in self.filter_ids else mask[pos] for pos, token_id in enumerate(ids[1:])]
                for ids, mask in zip(decoded_ids, special_tokens_masks)
            ]
            # breakpoint()
        
        layer_id = len(prompt_hidden_states) - 1
   
        # for layer_id in range(len(prompt_hidden_states)):
        embeddings["avg_ppt_layer"] = []
        embeddings["fst_gen_layer"] = []
        embeddings["avg_all_layer"] = []
        embeddings["avg_gen_layer"] = []
        embeddings["last_gen_layer"] = []

        for batch_id in range(len(prompt_hidden_states[layer_id])):
                # it is left padding
                prompt_hidden_states_remove_pad = prompt_hidden_states[layer_id][batch_id, attention_mask[batch_id].bool(), :]

                special_tokens_mask = special_tokens_masks[batch_id]
                if sum(special_tokens_mask) < len(special_tokens_mask):
                    # breakpoint()
                    generation_hidden_states = torch.cat([hs[layer_id][batch_id] for hs, special_token in zip(hidden_states[1:], special_tokens_mask) if not special_token], dim=0)
                else:
                    # if all of them are special token, then there is a problem
                    generation_hidden_states = hidden_states[1][layer_id][batch_id]

                all_hidden_states = torch.cat([prompt_hidden_states_remove_pad, generation_hidden_states], dim=0)

                # embeddings["avg_ppt_layer"].append(prompt_hidden_states_remove_pad.mean(0).cpu())
                embeddings["avg_ppt_layer"].append(prompt_hidden_states_remove_pad[:-1, :].mean(0).cpu())
                embeddings["fst_gen_layer"].append(prompt_hidden_states_remove_pad[-1, :].cpu())
                embeddings["avg_all_layer"].append(all_hidden_states.mean(0).cpu())
                # embeddings["avg_gen_layer")].append(generation_hidden_states.mean(0).cpu())
                embeddings["avg_gen_layer"].append(torch.cat([prompt_hidden_states_remove_pad[-1:, :], generation_hidden_states], dim=0).mean(0).cpu())
                embeddings["last_gen_layer"].append(generation_hidden_states[-1, :].cpu())
            
        embeddings["avg_ppt_layer"] = torch.stack(embeddings["avg_ppt_layer"], dim=0)
        embeddings["fst_gen_layer"] = torch.stack(embeddings["fst_gen_layer"], dim=0)
        embeddings["avg_all_layer"] = torch.stack(embeddings["avg_all_layer"], dim=0)
        embeddings["avg_gen_layer"] = torch.stack(embeddings["avg_gen_layer"], dim=0)
        embeddings["last_gen_layer"] = torch.stack(embeddings["last_gen_layer"], dim=0)

        return embeddings


    @classmethod
    def load(cls, model_args, **kwargs):
        # Loading the base model
        """
        base_model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True)
        """

        encoder = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", 
            torch_dtype="auto",
            device_map="cuda" if torch.cuda.is_available() else "cpu",                                                              
            low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", padding_side='left', truncation_side="left")

        model = cls(
                encoder=encoder,
                tokenizer=tokenizer,
                pooling=model_args.pooling,
                normalize=True,
                temperature=model_args.temperature,
            )

        return model


    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def forward(self, qry: dict[str, Tensor] = None, tgt: dict[str, Tensor] = None, *args, **kwargs):
        qry_reps = self.encode_input(qry) if qry else None  # (bsz_per_device, dim)
        tgt_reps = self.encode_input(tgt) if tgt else None # (bsz_per_device, dim)
        return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}


    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
