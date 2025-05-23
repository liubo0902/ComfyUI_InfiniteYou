# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from torch import Tensor, nn
from einops import rearrange, repeat

import comfy
from comfy.controlnet import controlnet_config, controlnet_load_state_dict, ControlNet, StrengthType
from comfy.ldm.flux.model import Flux
from comfy.ldm.flux.layers import (timestep_embedding)
import comfy.ldm.common_dit

class InfuseNet(ControlNet):
    def __init__(self, 
                 control_model=None, 
                 id_embedding = None,
                 global_average_pooling=False, 
                 compression_ratio=8, 
                 latent_format=None, 
                 load_device=None, 
                 manual_cast_dtype=None, 
                 extra_conds=["y"], 
                 strength_type=StrengthType.CONSTANT, 
                 concat_mask=False, 
                 preprocess_image=lambda a: a):
        super().__init__(control_model=control_model, 
                         global_average_pooling=global_average_pooling, 
                         compression_ratio=compression_ratio, 
                         latent_format=latent_format, 
                         load_device=load_device, 
                         manual_cast_dtype=manual_cast_dtype, 
                         extra_conds=extra_conds, 
                         strength_type=strength_type, 
                         concat_mask=concat_mask, 
                         preprocess_image=preprocess_image)
        self.id_embedding = id_embedding

    def copy(self):
        c = InfuseNet(None, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        c.control_model = self.control_model
        c.control_model_wrapped = self.control_model_wrapped
        c.id_embedding = self.id_embedding
        self.copy_to(c)
        return c
    
    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        cond = cond.copy()
        cond['crossattn_controlnet'] = self.id_embedding
        cond['c_crossattn'] = self.id_embedding
        return super().get_control(x_noisy, t, cond, batched_number, transformer_options)
    
class InfuseNetFlux(Flux):
    def __init__(self, latent_input=False, num_union_modes=0, mistoline=False, control_latent_channels=None, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__(final_layer=False, dtype=dtype, device=device, operations=operations, **kwargs)

        self.main_model_double = 19
        self.main_model_single = 38

        self.mistoline = mistoline
        # add ControlNet blocks
        if self.mistoline:
            control_block = lambda : MistolineControlnetBlock(self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            control_block = lambda : operations.Linear(self.hidden_size, self.hidden_size, dtype=dtype, device=device)

        self.controlnet_blocks = nn.ModuleList([])
        for _ in range(self.params.depth):
            self.controlnet_blocks.append(control_block())

        self.controlnet_single_blocks = nn.ModuleList([])
        for _ in range(self.params.depth_single_blocks):
            self.controlnet_single_blocks.append(control_block())

        self.num_union_modes = num_union_modes
        self.controlnet_mode_embedder = None
        if self.num_union_modes > 0:
            self.controlnet_mode_embedder = operations.Embedding(self.num_union_modes, self.hidden_size, dtype=dtype, device=device)

        self.gradient_checkpointing = False
        self.latent_input = latent_input
        if control_latent_channels is None:
            control_latent_channels = self.in_channels
        else:
            control_latent_channels *= 2 * 2 #patch size

        self.pos_embed_input = operations.Linear(control_latent_channels, self.hidden_size, bias=True, dtype=dtype, device=device)
        if not self.latent_input:
            if self.mistoline:
                self.input_cond_block = MistolineCondDownsamplBlock(dtype=dtype, device=device, operations=operations)
            else:
                self.input_hint_block = nn.Sequential(
                    operations.Conv2d(3, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, stride=2, dtype=dtype, device=device),
                    nn.SiLU(),
                    operations.Conv2d(16, 16, 3, padding=1, dtype=dtype, device=device)
                )

    def forward_orig(
        self,
        img: Tensor,
        img_ids: Tensor,
        controlnet_cond: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control_type: Tensor = None,
        out_mask: Tensor = None
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)

        controlnet_cond = self.pos_embed_input(controlnet_cond)
        img = img + controlnet_cond
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        if self.controlnet_mode_embedder is not None and len(control_type) > 0:
            control_cond = self.controlnet_mode_embedder(torch.tensor(control_type, device=img.device), out_dtype=img.dtype).unsqueeze(0).repeat((txt.shape[0], 1, 1))
            txt = torch.cat([control_cond, txt], dim=1)
            txt_ids = torch.cat([txt_ids[:,:1], txt_ids], dim=1)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        controlnet_double = ()

        for i in range(len(self.double_blocks)):
            img, txt = self.double_blocks[i](img=img, txt=txt, vec=vec, pe=pe)
            controlnet_double = controlnet_double + (self.controlnet_blocks[i](img),)

        img = torch.cat((txt, img), 1)

        controlnet_single = ()

        for i in range(len(self.single_blocks)):
            img = self.single_blocks[i](img, vec=vec, pe=pe)
            controlnet_single = controlnet_single + (self.controlnet_single_blocks[i](img[:, txt.shape[1] :, ...]),)

        repeat = math.ceil(self.main_model_double / len(controlnet_double))
        if self.latent_input:
            out_input = ()
            for x in controlnet_double:
                if out_mask is not None:
                    out_input += (x * out_mask,) * repeat
                else:
                    out_input += (x,) * repeat
        else:
            out_input = (controlnet_double * repeat)

        out = {"input": out_input[:self.main_model_double]}
        if len(controlnet_single) > 0:
            repeat = math.ceil(self.main_model_single / len(controlnet_single))
            out_output = ()
            if self.latent_input:
                for x in controlnet_single:
                    if out_mask is not None:
                        out_output += (x * out_mask,) * repeat
                    else:
                        out_output += (x,) * repeat
            else:
                out_output = (controlnet_single * repeat)
            out["output"] = out_output[:self.main_model_single]
        return out

    def forward(self, x, timesteps, context, y, guidance=None, hint=None, **kwargs):
        patch_size = 2
        if self.latent_input:
            hint = comfy.ldm.common_dit.pad_to_patch_size(hint, (patch_size, patch_size))
        elif self.mistoline:
            hint = hint * 2.0 - 1.0
            hint = self.input_cond_block(hint)
        else:
            hint = hint * 2.0 - 1.0
            hint = self.input_hint_block(hint)

        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        bs, c, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        control_mask = kwargs.get("control_mask", None)
        out_mask = None
        if control_mask is not None:
            in_mask = comfy.sampler_helpers.prepare_mask(control_mask, (bs, c, h, w), img.device)
            in_mask = comfy.ldm.common_dit.pad_to_patch_size(in_mask, (patch_size, patch_size))
            in_mask = rearrange(in_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
            # (b, seq_len, _) =>(b, seq_len, pulid.dim)
            in_mask = in_mask[..., 0].unsqueeze(-1).repeat(1, 1, img.shape[-1]).to(dtype=img.dtype)
            img = img * in_mask
            
            out_mask = comfy.sampler_helpers.prepare_mask(control_mask, (bs, 
                                                                      self.hidden_size // (patch_size * patch_size), h, w), img.device)
            out_mask = comfy.ldm.common_dit.pad_to_patch_size(out_mask, (patch_size, patch_size))
            out_mask = rearrange(out_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
        return self.forward_orig(img, img_ids, hint, context, txt_ids, timesteps, y, guidance, control_type=kwargs.get("control_type", []), out_mask=out_mask)

def load_infuse_net_flux(ckpt_path, model_options={}):
    sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    new_sd = comfy.model_detection.convert_diffusers_mmdit(sd, "")
    model_config, operations, load_device, unet_dtype, manual_cast_dtype, offload_device = controlnet_config(new_sd, model_options=model_options)
    for k in sd:
        new_sd[k] = sd[k]

    num_union_modes = 0
    union_cnet = "controlnet_mode_embedder.weight"
    if union_cnet in new_sd:
        num_union_modes = new_sd[union_cnet].shape[0]

    control_latent_channels = new_sd.get("pos_embed_input.weight").shape[1] // 4
    concat_mask = False
    if control_latent_channels == 17:
        concat_mask = True

    control_model = InfuseNetFlux(latent_input=True, num_union_modes=num_union_modes, control_latent_channels=control_latent_channels, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.Flux()
    extra_conds = ['y', 'guidance']
    control = InfuseNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control
