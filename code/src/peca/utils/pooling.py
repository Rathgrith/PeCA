import torch
import torch.nn.functional as F


def segment_pooling(feats_maps, seg_images, seg_num, pool_size):
    B, S, C, H_feat, W_feat = feats_maps.shape
    _, _, H_seg, W_seg = seg_images.shape
    L = int(seg_num.max().item())

    pooled_seg_feats = torch.zeros((B, S, L, C), device=feats_maps.device)
    pool_k_h = max(1, H_seg // pool_size[0])
    pool_k_w = max(1, W_seg // pool_size[1])
    H_out = H_seg // pool_k_h
    W_out = W_seg // pool_k_w

    skip_resize = (H_feat == H_out) and (W_feat == W_out)

    def safe_interpolate(x, size):
        """
        Avoid int32 overflow when B*C*H*W is large by chunking over batch.
        """
        int_max = 2_147_483_647
        B_local, C_local = x.shape[0], x.shape[1]
        max_batch = max(1, int(int_max // (C_local * size[0] * size[1])))
        if max_batch >= B_local:
            return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

        outputs = []
        for start in range(0, B_local, max_batch):
            end = min(B_local, start + max_batch)
            outputs.append(F.interpolate(x[start:end], size=size, mode="bilinear", align_corners=False))
        return torch.cat(outputs, dim=0)

    for s in range(S):
        seg_image = seg_images[:, s]  # [B, H, W]
        feats_map = feats_maps[:, s]  # [B, C, H_feat, W_feat]
        max_seg_num = int(seg_num[:, s].max().item())

        masks = torch.stack([(seg_image == (i + 1)) for i in range(max_seg_num)])  # [L, B, H, W]
        masks = masks.permute(1, 0, 2, 3).reshape(B * max_seg_num, 1, H_seg, W_seg).float()

        masks_pooled = F.max_pool2d(masks, kernel_size=(pool_k_h, pool_k_w))  # [B*L, 1, H', W']
        masks_pooled = masks_pooled.view(B, max_seg_num, H_out, W_out).bool()  # [B, L, H', W']

        if not skip_resize:
            feats_map = safe_interpolate(feats_map, size=(H_out, W_out))

        # flatten
        feats_flat = feats_map.view(B, C, -1)  # [B, C, H'*W']
        masks_flat = masks_pooled.view(B, max_seg_num, -1).float()  # [B, L, H'*W']

        # avoid divide-by-zero
        mask_sum = masks_flat.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [B, L, 1]

        pooled = torch.matmul(masks_flat, feats_flat.transpose(1, 2)) / mask_sum  # [B, L, C]
        pooled_seg_feats[:, s, :max_seg_num] = pooled

    return pooled_seg_feats
