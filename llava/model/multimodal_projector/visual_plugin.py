import torch
import torch.nn as nn
from functools import partial
from timm.models.regnet import RegStage
from timm.layers import LayerNorm2d
from transformers.models.deformable_detr.modeling_deformable_detr import (
    DeformableDetrDecoder,
    DeformableDetrDecoderLayer,
    DeformableDetrDecoderOutput,
)
from torch.nn.init import trunc_normal_
from einops import rearrange
import numpy

def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()

    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape

def cal_rel_pos_spatial(
    attn,
    q,
    has_cls_embed,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Abstractor(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_pre_layers=3, 
                 num_post_layers=3, 
                 pool_stride=2, 
                 kernel_size=3,
                 rel_pos_spatial=False,
                 grouping=''):
        super(Abstractor, self).__init__()
        self.type = grouping.split('_')[0] # option: cabstractor, dabstractor
        self.is_gate = grouping.find('gate')!=-1
        
        if self.is_gate:
            self.pooler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.gate = nn.Parameter(torch.tensor([0.0]))

        if self.type == 'cabstractor':
            RegBlock = partial(
                RegStage,
                stride=1,
                dilation=1,
                act_layer=nn.SiLU,
                norm_layer=LayerNorm2d,
            )
            s1 = RegBlock(
                num_pre_layers,
                hidden_dim,
                hidden_dim,
            )
            s2 = RegBlock(
                num_post_layers,
                hidden_dim,
                hidden_dim,
            )
            sampler = nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride)
            self.net = nn.Sequential(s1, sampler, s2)
        elif self.type == 'dabstractor':
            self.net = nn.Identity()
        elif self.type == 'DWConvabstractor':
            depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=pool_stride+1, stride=pool_stride, padding=pool_stride//2, groups=hidden_dim, bias=False)
            norm = LayerNorm2d(hidden_dim)
            act = nn.GELU()
            self.net = nn.Sequential(depthwise, norm, act)
        elif self.type == 'DWKSabstractor':
            depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=pool_stride, stride=pool_stride, padding=0, groups=hidden_dim, bias=False)
            norm = LayerNorm2d(hidden_dim)
            act = nn.GELU()
            self.net = nn.Sequential(depthwise, norm, act)
        elif self.type == 'MSAabstractor':
            msa = MultiScaleAttention(dim=hidden_dim, 
                                      dim_out=hidden_dim, 
                                      input_size=(24,24),
                                      kernel_q=(kernel_size,kernel_size),
                                      stride_q=(pool_stride,pool_stride),
                                      rel_pos_spatial=rel_pos_spatial,
            )
            norm = LayerNorm2d(hidden_dim)
            self.net = nn.Sequential(msa, norm)
            self.pooler = nn.AvgPool2d(kernel_size=kernel_size, stride=pool_stride, padding= kernel_size // 2)
        else:
            self.net = nn.Identity()



    def forward(self,x):
        if self.is_gate:
            x = self.net(x) * self.gate.tanh() + self.pooler(x)
        else:
            x = self.net(x) + self.pooler(x)
        return x

class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        input_size,
        num_heads=8,
        qkv_bias=False,
        kernel_q=(3, 3),
        kernel_kv=(1, 1),
        stride_q=(2, 2),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=False,
        mode="avg",
        pool_first=True,
        rel_pos_spatial=True,
        rel_pos_zero_init=False,
        residual_pooling=True,
    ):
        super().__init__()
        self.pool_first = pool_first

        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim**-0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = (
                pool_op(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv2d(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        # relative pos embedding
        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, hw_shape=(24,24)):
        x_dim = len(x.shape)
        if len(x.shape) == 4:
            # flatten HW dims into one
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W)
        x = x.permute(0, 2, 1)
        B, N, _ = x.shape

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = (
                self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_q", None),
        )
        k, k_shape = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_k", None),
        )
        v, v_shape = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=getattr(self, "norm_v", None),
        )

        if self.pool_first:
            q_N = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
            k_N = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
            v_N = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)

            q = q.permute(0, 2, 1, 3).reshape(B, q_N, -1)
            q = self.q(q).reshape(B, q_N, self.num_heads, -1).permute(0, 2, 1, 3)

            v = v.permute(0, 2, 1, 3).reshape(B, v_N, -1)
            v = self.v(v).reshape(B, v_N, self.num_heads, -1).permute(0, 2, 1, 3)

            k = k.permute(0, 2, 1, 3).reshape(B, k_N, -1)
            k = self.k(k).reshape(B, k_N, self.num_heads, -1).permute(0, 2, 1, 3)

        N = q.shape[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(
                attn,
                q,
                self.has_cls_embed,
                q_shape,
                k_shape,
                self.rel_pos_h,
                self.rel_pos_w,
            )

        attn = attn.softmax(dim=-1)
        x = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x = x + q

        x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        
        if x_dim == 4:
            x = x.reshape(B, q_shape[0], q_shape[1], -1).permute(0, 3, 1, 2)

        return x
    

class DAbstractor(nn.Module):
    def __init__(self,config, num_feature_levels,decoder_layers ):
        super(Abstractor, self).__init__()
        self.num_feature_levels = num_feature_levels
        self.layers = nn.ModuleList(
            [DeformableDetrDecoderLayer(config) for _ in range(decoder_layers)]
        )
    def _get_query_reference_points(self, spatial_shapes, valid_ratios):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        steps = int(self.num_queries**0.5)
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, height - 0.5, steps, dtype=torch.float32),
                torch.linspace(0.5, width - 0.5, steps, dtype=torch.float32),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points.squeeze(2)

    def _forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        intermediate = ()
        intermediate_reference_points = ()

        for _, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    encoder_hidden_states=encoder_hidden_states,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return DeformableDetrDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _process_v_features(self, visual_feat):
        # visual_feat: [B, len, dim] or [B, lvls, len, dim]
        if self.except_cls:
            visual_feat = visual_feat[:, :, 1:] if self.isMs else visual_feat[:, 1:]

        if self.isMs:
            visual_feats = []
            for level in range(self.num_feature_levels):
                visual_feats.append(self.input_proj[level](visual_feat[:, level]))
            visual_feat = torch.stack(visual_feats, 1)

            # add pos emb [1, len, dim]
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb.unsqueeze(1)

            # add lvl emb [1, lvls, 1, dim]
            visual_feat = visual_feat + self.level_emb
            visual_feat = visual_feat.flatten(1, 2)  # [B, lvls, v_len, dim] -> [B, lvls*v_len, dim]
        else:
            visual_feat = self.input_proj[0](visual_feat)
            if self.v_pos_emb is not None:
                visual_feat = visual_feat + self.v_pos_emb

        return visual_feat

    def _convert_dtype_device(self, tgt_feat, dtype=None, device=None):
        # tgt_feat: target tensor to be converted
        _dtype = tgt_feat.dtype if dtype is None else dtype
        _device = tgt_feat.device if device is None else device

        tgt_feat = tgt_feat.type(_dtype).to(_device)

        return tgt_feat

    def _prepare_ddetr_inputs(self, batch_size, seq_len, lvls, dtype=None, device=None):
        # assume there are no paddings in a feature map
        valid_ratios = torch.ones(batch_size, lvls, 2)

        # assume all feature maps have the same sequence length (i.e., the same shape)
        spatial_shapes = torch.tensor([int(seq_len**0.5), int(seq_len**0.5)]).repeat(lvls, 1)
        level_start_index = torch.arange(0, seq_len * lvls, seq_len)

        if dtype is not None and device is not None:
            valid_ratios = self._convert_dtype_device(valid_ratios, dtype=dtype, device=device)
            spatial_shapes = self._convert_dtype_device(
                spatial_shapes, dtype=torch.long, device=device
            )
            level_start_index = self._convert_dtype_device(
                level_start_index, dtype=torch.long, device=device
            )

        return valid_ratios, spatial_shapes, level_start_index

    def _make_pooled_queries(self, visual_feat):
        assert (
            self.num_feature_levels == 1
        )  # currently do not support multi-scale features for the v-pooled Q

        batch_size, seq_len, h_dim = visual_feat.shape
        query_embeds = self.query_position_embeddings.weight
        if self.pooled_v_target != "none":
            hw_v = int(seq_len**0.5)
            hw_q = int(self.num_queries**0.5)
            visual_feat = rearrange(visual_feat, "b (h w) d -> b d h w", h=hw_v, w=hw_v)
            if self.pooled_v_target == "tgt":
                query_embed = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                target = self.downsampler(visual_feat)
                target = rearrange(target, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
            else:
                target = query_embeds.unsqueeze(0).expand(batch_size, -1, -1)
                query_embed = self.downsampler(visual_feat)
                query_embed = rearrange(query_embed, "b d h w -> b (h w) d", h=hw_q, w=hw_q)
        else:
            query_embed, target = torch.split(query_embeds, h_dim, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)
            target = target.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embed, target

    def forward(self, visual_feat):
        
        # deformable attention only supports fp32
        original_dtype = visual_feat.type()
        visual_feat = visual_feat.type(torch.cuda.FloatTensor)
        visual_feat = self._process_v_features(visual_feat)

        batch_size, seq_len, h_dim = visual_feat.shape
        seq_len /= self.num_feature_levels

        query_embed, target = self._make_pooled_queries(visual_feat)
        reference_points = self.reference_points.expand(batch_size, -1, -1)

        valid_ratios, spatial_shapes, level_start_index = self._prepare_ddetr_inputs(
            batch_size, seq_len, self.num_feature_levels, visual_feat.dtype, visual_feat.device
        )

        decoder_outputs_dict = self._forward(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=visual_feat,
            valid_ratios=valid_ratios,
            reference_points=reference_points,
            return_dict=True,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        decoder_outputs = decoder_outputs_dict.last_hidden_state

        # if self.eos_tokens is not None:
        #     decoder_outputs = torch.cat(
        #         [decoder_outputs, self.eos_tokens.expand(batch_size, -1, -1)], dim=1
        #     )

        # decoder_outputs = self.output_proj(decoder_outputs)
        decoder_outputs = decoder_outputs.type(original_dtype)
        return decoder_outputs
    

if __name__ == '__main__':
    model = Abstractor(hidden_dim=1024, kernel_size=3, pool_stride=2,rel_pos_spatial=True,grouping='MSAabstractor_gate')
    input = torch.randn(10,1024, 24,24 )

    output = model(input)
    print(output.shape)
        