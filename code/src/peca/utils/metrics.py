import torch


def calculate_accuracy(nearest_patch_indices, seg_colors_src, seg_colors_tgt, seg_sizes_tgt):
    def is_background(color):
        return color[3] <= 0

    total_seg_num = 0
    correct_seg_num = 0
    total_pix_num = 0
    correct_pix_num = 0
    total_seg_thres_num = 0
    correct_seg_thres_num = 0
    total_fg_pix_num = 0
    correct_fg_pix_num = 0
    pix_bg_miou_numerator = 0
    pix_bg_miou_denominator = 0
    seg_bg_miou_numerator = 0
    seg_bg_miou_denominator = 0

    for idx, color_index in enumerate(nearest_patch_indices):
        gt_color = seg_colors_tgt[idx]
        pred_color = seg_colors_src[color_index]
        size = seg_sizes_tgt[idx]
        seg_size = int(size.item()) if isinstance(size, torch.Tensor) else int(size)
        total_pix_num += seg_size
        total_seg_num += 1
        is_bg_gt = is_background(gt_color)
        is_bg_pred = is_background(pred_color)

        if seg_size > 10:
            total_seg_thres_num += 1
        if not is_bg_gt:
            total_fg_pix_num += seg_size

        if torch.equal(pred_color, gt_color):
            correct_pix_num += seg_size
            correct_seg_num += 1
            if seg_size > 10:
                correct_seg_thres_num += 1
            if is_bg_gt:
                pix_bg_miou_numerator += seg_size
                pix_bg_miou_denominator += seg_size
                seg_bg_miou_numerator += 1
                seg_bg_miou_denominator += 1
            else:
                correct_fg_pix_num += seg_size
        elif is_bg_gt or is_bg_pred:
            pix_bg_miou_denominator += seg_size
            seg_bg_miou_denominator += 1

    pix_acc = correct_pix_num / total_pix_num if total_pix_num else 0
    seg_acc = correct_seg_num / total_seg_num if total_seg_num else 0
    seg_acc_thres = correct_seg_thres_num / total_seg_thres_num if total_seg_thres_num else 0
    pix_fg_acc = correct_fg_pix_num / total_fg_pix_num if total_fg_pix_num else 0
    pix_bg_acc = pix_bg_miou_numerator / pix_bg_miou_denominator if pix_bg_miou_denominator else 0
    seg_bg_miou = seg_bg_miou_numerator / seg_bg_miou_denominator if seg_bg_miou_denominator else 0
    return seg_acc, seg_acc_thres, pix_acc, pix_fg_acc, pix_bg_acc, seg_bg_miou
