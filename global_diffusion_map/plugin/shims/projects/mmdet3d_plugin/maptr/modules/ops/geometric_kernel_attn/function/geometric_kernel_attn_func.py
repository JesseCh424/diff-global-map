import torch


class GeometricKernelAttentionFunc:
    @staticmethod
    def apply(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        """
        Pure PyTorch fallback of geometric kernel attention.
        Shapes expected:
        - value: (bs, num_value, num_heads, dim)
        - value_spatial_shapes: (num_levels, 2) with (H, W)
        - value_level_start_index: (num_levels,)
        - sampling_locations: (bs, num_query, num_heads, num_levels, num_all_points, 2)
        - attention_weights: (bs, num_query, num_heads, num_levels, num_all_points)
        Returns (bs, num_query, embed_dims) where embed_dims = num_heads * dim
        """
        device = value.device
        dtype = value.dtype
        bs, num_value, num_heads, dim = value.shape
        bs2, num_query, num_heads2, num_levels, num_pts_all, _ = sampling_locations.shape
        assert bs == bs2 and num_heads == num_heads2, 'Mismatched shapes'

        # value -> (bs, num_heads, num_value, dim)
        value_perm = value.permute(0, 2, 1, 3).contiguous()

        # Spatial shapes and starts
        spatial = value_spatial_shapes.to(device=device)
        starts = value_level_start_index.to(device=device)

        # Clamp and compute flattened indices per level
        # sampling_locations are expected integer grid coords
        sl = sampling_locations.to(device=device)
        sl = torch.round(sl).long()

        # Build per-level width tensor for broadcasting
        # spatial: (num_levels, 2) -> W per level
        widths = spatial[:, 1].to(device=device)

        # Prepare indices tensor
        idx = sl[..., 0].clone()  # x
        idy = sl[..., 1].clone()  # y
        # Clamp per level
        for l in range(num_levels):
            H = int(spatial[l, 0].item())
            W = int(spatial[l, 1].item())
            idx[:, :, :, l, :].clamp_(min=0, max=W - 1)
            idy[:, :, :, l, :].clamp_(min=0, max=H - 1)

        # Compute flattened index: start[l] + x + y * W_l
        # Broadcast starts and widths
        starts_b = starts.view(1, 1, 1, num_levels, 1).expand(bs, num_query, num_heads, num_levels, num_pts_all)
        widths_b = widths.view(1, 1, 1, num_levels, 1).expand(bs, num_query, num_heads, num_levels, num_pts_all)
        flat_idx = starts_b + idx + idy * widths_b
        flat_idx = flat_idx.long()

        # Gather values
        # value_perm: (bs, num_heads, num_value, dim)
        # reshape for gather: (bs*num_heads, num_value, dim)
        vh = value_perm.reshape(bs * num_heads, num_value, dim)
        # indices to (bs*num_heads, num_query*num_levels*num_pts_all)
        flat_idx_h = flat_idx.permute(0, 2, 1, 3, 4).reshape(bs * num_heads, -1)
        gathered = vh.gather(1, flat_idx_h.unsqueeze(-1).expand(-1, -1, dim))
        gathered = gathered.view(bs, num_heads, num_query, num_levels * num_pts_all, dim)

        # Attention weights
        aw = attention_weights.permute(0, 2, 1, 3, 4).contiguous().view(bs, num_heads, num_query, num_levels * num_pts_all, 1)

        out = (gathered * aw).sum(dim=3)  # (bs, num_heads, num_query, dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, num_query, num_heads * dim)
        return out.to(dtype)

