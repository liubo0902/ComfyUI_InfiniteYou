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

import comfy

from comfy.controlnet import controlnet_config, controlnet_load_state_dict, ControlNet, StrengthType

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

    control_model = comfy.ldm.flux.controlnet.ControlNetFlux(latent_input=True, num_union_modes=num_union_modes, control_latent_channels=control_latent_channels, operations=operations, device=offload_device, dtype=unet_dtype, **model_config.unet_config)
    control_model = controlnet_load_state_dict(control_model, new_sd)

    latent_format = comfy.latent_formats.Flux()
    extra_conds = ['y', 'guidance']
    control = InfuseNet(control_model, compression_ratio=1, latent_format=latent_format, concat_mask=concat_mask, load_device=load_device, manual_cast_dtype=manual_cast_dtype, extra_conds=extra_conds)
    return control
