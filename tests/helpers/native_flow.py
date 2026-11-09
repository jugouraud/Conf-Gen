"""Deterministic SD3-shaped encoder/transformer boundary without model downloads."""

import torch


class TinyTransformer(torch.nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *, hidden_states, encoder_hidden_states, **kwargs):
        self.calls += 1
        condition = encoder_hidden_states.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return (hidden_states * 0.01 + condition,)


class TinyNativePipe:
    def __init__(self):
        self.transformer = TinyTransformer()
        self.text_encoder = torch.nn.Identity()
        self.text_encoder_2 = torch.nn.Identity()
        self.text_encoder_3 = torch.nn.Identity()
        self.vae = torch.nn.Identity()

    def remove_all_hooks(self):
        pass

    def maybe_free_model_hooks(self):
        pass

    def _condition(self, prompt):
        if not prompt:
            return 0.0
        return (0.0 if prompt.startswith("legal") else 10.0) + int(
            prompt.rsplit(" ", 1)[1]
        ) / 100

    def _get_clip_prompt_embeds(self, *, prompt, device, clip_model_index):
        value = self._condition(prompt)
        return torch.full((1, 1, 1), value, device=device), torch.full(
            (1, 1), value, device=device
        )

    def _get_t5_prompt_embeds(self, *, prompt, device, max_sequence_length):
        return torch.full((1, 1, 2), self._condition(prompt), device=device)

    def encode_prompt(self, *, prompt, device, **kwargs):
        first, pooled_first = self._get_clip_prompt_embeds(
            prompt=prompt, device=device, clip_model_index=0
        )
        second, pooled_second = self._get_clip_prompt_embeds(
            prompt=prompt, device=device, clip_model_index=1
        )
        t5 = self._get_t5_prompt_embeds(
            prompt=prompt, device=device, max_sequence_length=256
        )
        return (
            torch.cat((torch.cat((first, second), dim=-1), t5), dim=1),
            None,
            torch.cat((pooled_first, pooled_second), dim=-1),
            None,
        )
