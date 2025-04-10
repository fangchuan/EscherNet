from transformers import ConvNextV2Model
import torch
from typing import Optional
import einops

class CN_encoder(ConvNextV2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            pixel_values: torch.FloatTensor = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print(f'pixel_values: {pixel_values.shape}')
        # print(f'output_hidden_states: {output_hidden_states}')
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        embedding_output = self.embeddings(pixel_values)
        # print(f'embedding_output: {embedding_output.shape}')
        encoder_outputs = self.encoder(
            embedding_output,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # print(f'encoder_outputs: {encoder_outputs[0].shape}')
        last_hidden_state = encoder_outputs[0]
        image_embeddings = einops.rearrange(last_hidden_state, 'b c h w -> b (h w) c')
        # print(f'image_embeddings: {image_embeddings.shape}')
        image_embeddings = self.layernorm(image_embeddings)

        return image_embeddings