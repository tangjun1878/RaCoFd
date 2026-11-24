"""RaCoFD paper losses for the exact uploaded RT-DETRv4 codebase.

Three independent contributions are implemented here:
  (1) existing RT-DETRv4 DINOv3 F5 loss, unmodified via super().forward;
  (2) detector teacher's GT-matched classification AND box response KD, with
      distinct correctness/localization reliability;
  (3) GT-anchored DINOv3 instance feature distillation with a non-suppressive
      error-conditioned complementarity coefficient. This coefficient requires
      VFM flip-view stability and local GT-gradient agreement on SHARED F5.

The teachers, optional features and evidence computations are train-time only.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from ..core import register
from ..rtv4.rtv4_criterion import RTv4Criterion
from ..rtv4.box_ops import box_cxcywh_to_xyxy


# Constant diagnostics across every batch and distributed rank. MetricLogger
# synchronizes all meter keys collectively; variable keys can deadlock DDP
# when one rank happens to receive an image without GT objects.
_RACOFD_DIAG_KEYS = (
    'racofd_gt_objects_per_batch',
    'racofd_matched_gt_fraction',
    'racofd_teacher_cls_accuracy',
    'racofd_teacher_gt_prob',
    'racofd_teacher_box_iou',
    'racofd_student_box_iou',
    'racofd_student_cls_accuracy',
    'racofd_cls_kd_active_fraction',
    'racofd_box_kd_active_fraction',
    'racofd_new_kd_scale',
    'racofd_reliable_cls_fraction',
    'racofd_reliable_box_fraction',
    'racofd_inst_vfm_raw',
    'racofd_vfm_stability',
    'racofd_gt_grad_agree_cls',
    'racofd_gt_grad_agree_box',
    'racofd_complement_multiplier',
    'racofd_complement_active_fraction',
)


def aligned_iou_cxcywh(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Pairwise-aligned IoU for [N,4] normalized cxcywh (not full NxN IoU)."""
    a = box_cxcywh_to_xyxy(a.float())
    b = box_cxcywh_to_xyxy(b.float())
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    aa = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    bb = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (aa + bb - intersection + eps)


def aligned_giou_cxcywh(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6):
    """Aligned Generalized IoU, differentiable wrt student boxes."""
    a = box_cxcywh_to_xyxy(a.float())
    b = box_cxcywh_to_xyxy(b.float())
    lt = torch.maximum(a[:, :2], b[:, :2])
    rb = torch.minimum(a[:, 2:], b[:, 2:])
    wh = (rb-lt).clamp(min=0)
    intersection = wh.prod(-1)
    aa = (a[:, 2:]-a[:, :2]).clamp(min=0).prod(-1)
    bb = (b[:, 2:]-b[:, :2]).clamp(min=0).prod(-1)
    union = aa + bb - intersection
    iou = intersection / (union + eps)
    c_lt = torch.minimum(a[:, :2], b[:, :2])
    c_rb = torch.maximum(a[:, 2:], b[:, 2:])
    enclosure = (c_rb-c_lt).clamp(min=0).prod(-1)
    return iou - (enclosure-union)/(enclosure+eps)


def gt_rois(targets: List[dict], device: torch.device):
    """Normalized GT boxes -> [K,5] normalized xyxy, batch-index first."""
    pieces = []
    for batch_index, t in enumerate(targets):
        boxes = t['boxes']
        if boxes.numel() == 0:
            continue
        boxes_xyxy = box_cxcywh_to_xyxy(boxes.float()).clamp(0, 1)
        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        boxes_xyxy = boxes_xyxy[valid]
        if boxes_xyxy.shape[0]:
            ids = boxes_xyxy.new_full((len(boxes_xyxy), 1), batch_index)
            pieces.append(torch.cat([ids, boxes_xyxy], dim=1))
    if not pieces:
        return torch.empty((0, 5), device=device, dtype=torch.float32)
    return torch.cat(pieces, dim=0).to(device=device)


def normalized_rois_to_feature(rois: torch.Tensor, feature: torch.Tensor):
    h, w = feature.shape[-2:]
    feature_rois = rois.clone()
    feature_rois[:, [1, 3]] *= float(w)
    feature_rois[:, [2, 4]] *= float(h)
    return feature_rois


def pooled_gt_features(feature: torch.Tensor, rois: torch.Tensor, size: int = 2):
    if rois.shape[0] == 0:
        return feature.new_empty((0, feature.shape[1]))
    sampled = roi_align(feature.float(), normalized_rois_to_feature(rois, feature),
                        output_size=(size, size), spatial_scale=1.0, aligned=True)
    return sampled.mean(dim=(-1, -2))


def bernoulli_kl_from_logits(student_logits: torch.Tensor,
                             teacher_logits: torch.Tensor, temperature: float):
    """Per-target mean Bernoulli KL, rather than categorical softmax KL."""
    s = student_logits.float()/temperature
    t = teacher_logits.float()/temperature
    p = torch.sigmoid(t).clamp(1e-6, 1-1e-6)
    kl = (p * (torch.log(p) - F.logsigmoid(s)) +
          (1-p) * (torch.log1p(-p) - F.logsigmoid(-s)))
    return kl.mean(dim=-1) * (temperature**2)


@register()
class RaCoFDCriterion(RTv4Criterion):
    """Drop-in train criterion. Existing RTv4 inference/losses are preserved."""
    __share__ = ['num_classes']
    __inject__ = ['matcher']

    def __init__(self, matcher, weight_dict, losses,
                 alpha=0.2, gamma=2.0, num_classes=80, reg_max=32,
                 boxes_weight_format=None, share_matched_indices=False,
                 mal_alpha=None, use_uni_set=True, distill_adaptive_params=None,
                 racofd=None):
        super().__init__(matcher=matcher, weight_dict=weight_dict, losses=losses,
                         alpha=alpha, gamma=gamma, num_classes=num_classes,
                         reg_max=reg_max, boxes_weight_format=boxes_weight_format,
                         share_matched_indices=share_matched_indices,
                         mal_alpha=mal_alpha, use_uni_set=use_uni_set,
                         distill_adaptive_params=distill_adaptive_params)
        cfg = dict(racofd or {})
        self.use_detector_kd = bool(cfg.get('detector_kd', True))
        self.use_reliability_filtering = bool(cfg.get('reliability_filtering', True))
        # For a converged COCO student, X is not necessarily better on every GT.
        # Optional extra gate applies to response KD only, not the VFM evidence.
        self.student_relative_kd = bool(cfg.get('student_relative_kd', False))
        self.cls_advantage_margin = float(cfg.get('cls_advantage_margin', 0.02))
        self.box_advantage_margin = float(cfg.get('box_advantage_margin', 0.02))
        self.new_kd_warmup_epochs = float(cfg.get('new_kd_warmup_epochs', 0.0))
        if min(self.cls_advantage_margin, self.box_advantage_margin,
               self.new_kd_warmup_epochs) < 0:
            raise ValueError('RaCoFD advantage margins and warmup must be nonnegative')
        self.use_instance_vfm = bool(cfg.get('instance_vfm', True))
        self.use_grad_evidence = bool(cfg.get('gradient_evidence', True))
        self.racofd_flip_consistency = bool(cfg.get('flip_consistency', True))
        self.use_detector_feature = bool(cfg.get('detector_feature_kd', False))
        self.detector_feature_weight = float(cfg.get('detector_feature_weight', 0.1))
        self.cls_weight = float(cfg.get('cls_weight', 2.0))
        self.box_weight = float(cfg.get('box_weight', 2.0))
        self.instance_weight = float(cfg.get('instance_weight', 0.5))
        self.complement_alpha = float(cfg.get('complement_alpha', 1.0))
        self.cls_min_conf = float(cfg.get('cls_min_conf', 0.1))
        self.box_min_iou = float(cfg.get('box_min_iou', 0.3))
        self.temperature = float(cfg.get('temperature', 2.0))
        self.vfm_stability_temperature = float(cfg.get('stability_temperature', 0.5))
        self.grad_interval = int(cfg.get('gradient_interval', 4))
        self.grad_momentum = float(cfg.get('gradient_momentum', 0.9))
        self.max_extra_weight = float(cfg.get('max_extra_weight', 2.0))
        self.roi_output_size = int(cfg.get('roi_output_size', 2))
        if not 0 <= self.grad_momentum < 1 or self.grad_interval <= 0:
            raise ValueError('RaCoFD gradient_interval > 0 and 0 <= momentum < 1 required')
        if self.cls_weight < 0 or self.box_weight < 0 or self.instance_weight < 0:
            raise ValueError('RaCoFD weights must be nonnegative')
        self._ra_step = 0
        self.racofd_diagnostics = {}  # logging only; not part of loss_dict
        # Non-persistent class-specific sampled evidence for skipped steps.
        self.register_buffer('_ra_cls_prior', torch.zeros(num_classes), persistent=False)
        self.register_buffer('_ra_box_prior', torch.zeros(num_classes), persistent=False)

    @torch.no_grad()
    def _match_teacher_student(self, outputs: dict, teacher: dict, targets: List[dict]):
        # Both match independently to the SAME GT. Query IDs are never assumed equal.
        # The matcher implements DETR-style Hungarian matching from uploaded repository.
        src = {k: outputs[k] for k in ('pred_logits', 'pred_boxes')}
        dst = {k: teacher[k] for k in ('pred_logits', 'pred_boxes')}
        if src['pred_logits'].shape[0] != dst['pred_logits'].shape[0]:
            raise ValueError('Student and teacher batch sizes differ')
        if src['pred_logits'].shape[-1] != dst['pred_logits'].shape[-1]:
            raise ValueError('Student/teacher class counts differ: need compatible COCO heads')
        sm = self.matcher(src, targets)['indices']
        tm = self.matcher(dst, targets)['indices']
        batch, squery, tquery, gt_index = [], [], [], []
        for bi, ((sq, sg), (tq, tg)) in enumerate(zip(sm, tm)):
            teacher_map = {int(gi): int(qi) for qi, gi in zip(tq.tolist(), tg.tolist())}
            for qi, gi in zip(sq.tolist(), sg.tolist()):
                if gi in teacher_map:
                    batch.append(bi)
                    squery.append(qi)
                    tquery.append(teacher_map[gi])
                    gt_index.append(gi)
        dev = src['pred_logits'].device
        return tuple(torch.tensor(x, dtype=torch.long, device=dev)
                     for x in (batch, squery, tquery, gt_index))

    def _detection_kd(self, outputs, teacher, targets):
        dev = outputs['pred_logits'].device
        zero = outputs['pred_boxes'].sum() * 0.0
        reliab_cls = [torch.zeros(len(t['labels']), device=dev) for t in targets]
        reliab_box = [torch.zeros(len(t['labels']), device=dev) for t in targets]
        batch, squery, tquery, gtidx = self._match_teacher_student(outputs, teacher, targets)
        total_gt = sum(len(t['labels']) for t in targets)
        self.racofd_diagnostics['racofd_gt_objects_per_batch'] = float(total_gt)
        self.racofd_diagnostics['racofd_matched_gt_fraction'] = float(batch.numel() / max(total_gt, 1))
        if batch.numel() == 0:
            return {'loss_racofd_cls': zero, 'loss_racofd_box': zero}, reliab_cls, reliab_box

        sl = outputs['pred_logits'][batch, squery]
        tl = teacher['pred_logits'][batch, tquery].detach()
        sb = outputs['pred_boxes'][batch, squery]
        tb = teacher['pred_boxes'][batch, tquery].detach()
        gt_labels = torch.stack([targets[int(b)]['labels'][int(i)]
                                 for b, i in zip(batch.tolist(), gtidx.tolist())]).to(dev)
        gt_boxes = torch.stack([targets[int(b)]['boxes'][int(i)]
                                for b, i in zip(batch.tolist(), gtidx.tolist())]).to(dev)
        with torch.no_grad():
            p = torch.sigmoid(tl.float())
            gt_p = p.gather(1, gt_labels[:, None]).squeeze(1)
            rcls = (torch.argmax(tl, dim=-1) == gt_labels).float() * gt_p
            rcls = torch.where(gt_p >= self.cls_min_conf, rcls, rcls.new_zeros(()))
            true_cls_acc = (torch.argmax(tl, dim=-1) == gt_labels).float().mean()
            true_box_iou = aligned_iou_cxcywh(tb, gt_boxes).detach()
            rbox = torch.where(true_box_iou >= self.box_min_iou, true_box_iou, true_box_iou.new_zeros(()))
            self.racofd_diagnostics.update({
                'racofd_teacher_cls_accuracy': float(true_cls_acc.item()),
                'racofd_teacher_gt_prob': float(gt_p.mean().item()),
                'racofd_teacher_box_iou': float(true_box_iou.mean().item()),
                'racofd_reliable_cls_fraction': float((rcls > 0).float().mean().item()),
                'racofd_reliable_box_fraction': float((rbox > 0).float().mean().item()),
            })
            # True naive-dual ablation: no teacher-quality filtering.
            if not self.use_reliability_filtering:
                rcls = torch.ones_like(rcls)
                rbox = torch.ones_like(rbox)
            for n, (bi, gi) in enumerate(zip(batch.tolist(), gtidx.tolist())):
                reliab_cls[bi][gi] = rcls[n]
                reliab_box[bi][gi] = rbox[n]

            # Preserve the teacher-only reliability vectors above for EGCM.
            # A stronger student's box should not be pulled towards a worse X box.
            student_correct = sl.detach().argmax(dim=-1) == gt_labels
            student_gt_p = sl.detach().float().sigmoid().gather(1, gt_labels[:, None]).squeeze(1)
            student_iou = aligned_iou_cxcywh(sb.detach(), gt_boxes)
            cls_kd_weight, box_kd_weight = rcls, rbox
            if self.student_relative_kd and self.use_reliability_filtering:
                cls_advantage = (~student_correct) | (gt_p > student_gt_p + self.cls_advantage_margin)
                box_advantage = true_box_iou > student_iou + self.box_advantage_margin
                cls_kd_weight = rcls * cls_advantage.float()
                box_kd_weight = rbox * box_advantage.float()
            self.racofd_diagnostics.update({
                'racofd_student_box_iou': float(student_iou.mean().item()),
                'racofd_student_cls_accuracy': float(student_correct.float().mean().item()),
                'racofd_cls_kd_active_fraction': float((cls_kd_weight > 0).float().mean().item()),
                'racofd_box_kd_active_fraction': float((box_kd_weight > 0).float().mean().item()),
            })

        if self.use_detector_kd:
            per_class_kl = bernoulli_kl_from_logits(sl, tl, self.temperature)
            loss_cls = self.cls_weight * (per_class_kl * cls_kd_weight).sum() / max(len(batch), 1)
            per_box = (F.smooth_l1_loss(sb.float(), tb.float(), reduction='none').sum(-1)
                       + (1 - aligned_giou_cxcywh(sb, tb)))
            loss_box = self.box_weight * (per_box * box_kd_weight).sum() / max(len(batch), 1)
        else:
            loss_cls, loss_box = zero, zero
        return {'loss_racofd_cls': loss_cls, 'loss_racofd_box': loss_box}, reliab_cls, reliab_box

    def _task_gradient_agreement(self, losses: dict, local_loss: torch.Tensor,
                                 shared: torch.Tensor, rois: torch.Tensor,
                                 target_classes: torch.Tensor):
        """Sampled first-order task evidence on F5 *shared* with the detector.

        Autograd probes do not require higher-order derivatives, and coefficients
        are detached. Classification and box gradients are probed separately.
        """
        zero = local_loss.new_zeros(rois.shape[0])
        # RT-DETRv4 S uses MAL (not focal/softmax classification).
        cls_keys = ('loss_mal', 'loss_focal', 'loss_vfl')
        box_keys = ('loss_bbox', 'loss_giou')
        cls_loss = sum((losses[k] for k in cls_keys if k in losses), zero.new_zeros(()))
        box_loss = sum((losses[k] for k in box_keys if k in losses), zero.new_zeros(()))
        if not local_loss.requires_grad or not shared.requires_grad:
            return zero, zero

        def gradient(loss):
            if not loss.requires_grad:
                return None
            return torch.autograd.grad(loss, shared, allow_unused=True,
                                       retain_graph=True, create_graph=False)[0]

        grad_v = gradient(local_loss)
        grad_c = gradient(cls_loss)
        grad_b = gradient(box_loss)
        if grad_v is None:
            return zero, zero
        vg = pooled_gt_features(grad_v, rois, self.roi_output_size)

        def agreement(other):
            if other is None:
                return zero
            og = pooled_gt_features(other, rois, self.roi_output_size)
            return F.cosine_similarity(vg.float(), og.float(), dim=-1, eps=1e-8).clamp(min=0).detach()

        cls_agree = agreement(grad_c)
        box_agree = agreement(grad_b)

        with torch.no_grad():
            # Moving class-specific priors support intermittent gradient probes.
            for prior, values in ((self._ra_cls_prior, cls_agree),
                                  (self._ra_box_prior, box_agree)):
                counts = torch.zeros_like(prior)
                accum = torch.zeros_like(prior)
                counts.scatter_add_(0, target_classes, torch.ones_like(values))
                accum.scatter_add_(0, target_classes, values)
                valid = counts > 0
                new_values = accum / counts.clamp(min=1)
                prior[valid] = (self.grad_momentum * prior[valid] +
                                (1-self.grad_momentum) * new_values[valid])
        return cls_agree, box_agree

    def _instance_vfm(self, outputs, targets, teacher_vfm_flip_features,
                      cls_reliability, box_reliability, base_losses):
        sf = outputs.get('student_distill_output')
        tf = outputs.get('teacher_encoder_output')
        if sf is None or tf is None:
            raise RuntimeError('RaCoFD instance KD requires DINO teacher and student F5 projection')
        if sf.shape[1] != tf.shape[1]:
            raise ValueError(f'Instance VFM channel mismatch {sf.shape[1]} vs {tf.shape[1]}')
        rois = gt_rois(targets, sf.device)
        zero = sf.sum() * 0
        if rois.shape[0] == 0:
            return zero
        s_embed = pooled_gt_features(sf, rois, self.roi_output_size)
        t_embed = pooled_gt_features(tf.detach(), rois, self.roi_output_size)
        per_inst = 1.0 - F.cosine_similarity(s_embed.float(), t_embed.float(), dim=-1)
        if not self.use_instance_vfm:
            return zero

        if teacher_vfm_flip_features is not None:
            flipped = pooled_gt_features(teacher_vfm_flip_features.detach(), rois, self.roi_output_size)
            stable_dist = (F.normalize(t_embed.float(), dim=-1) -
                           F.normalize(flipped.float(), dim=-1)).square().sum(-1)
            stability = torch.exp(-stable_dist / max(self.vfm_stability_temperature, 1e-5))
        else:
            stability = per_inst.detach().new_ones(per_inst.shape)

        rcls = torch.cat(cls_reliability).detach()
        rbox = torch.cat(box_reliability).detach()
        # GT boxes excluded due to degenerate geometry may be present; standard
        # COCO sanitize removes them. Fail rather than silently misalign rows.
        if len(rcls) != len(per_inst):
            raise ValueError('Degenerate GT boxes remain; sanitize before RaCoFD instance KD')
        target_classes = torch.cat([t['labels'] for t in targets]).to(sf.device, dtype=torch.long)
        ac, ab = torch.zeros_like(rcls), torch.zeros_like(rbox)
        shared = outputs.get('racofd_shared_f5')
        self._ra_step += 1
        if self.use_grad_evidence and shared is not None and torch.is_grad_enabled():
            if (self._ra_step - 1) % self.grad_interval == 0:
                ac, ab = self._task_gradient_agreement(
                    base_losses, per_inst.mean(), shared, rois, target_classes)
            else:
                ac = self._ra_cls_prior[target_classes].detach()
                ab = self._ra_box_prior[target_classes].detach()

        evidence = stability.detach() * ((1-rcls)*ac + (1-rbox)*ab) / 2
        weights = (1 + self.complement_alpha * evidence.detach()).clamp(
            min=1, max=max(self.max_extra_weight, 1.0))
        self.racofd_diagnostics.update({
            'racofd_inst_vfm_raw': float(per_inst.detach().mean().item()),
            'racofd_vfm_stability': float(stability.detach().mean().item()),
            'racofd_gt_grad_agree_cls': float(ac.detach().mean().item()),
            'racofd_gt_grad_agree_box': float(ab.detach().mean().item()),
            'racofd_complement_multiplier': float(weights.detach().mean().item()),
            'racofd_complement_active_fraction': float((weights.detach() > 1.001).float().mean().item()),
        })
        return self.instance_weight * (per_inst * weights).mean()

    def _detector_feature_kd(self, outputs, teacher):
        sf = outputs.get('racofd_detection_f5')
        tf = teacher.get('feature_f5') if teacher is not None else None
        if sf is None or tf is None:
            raise RuntimeError('Detector feature KD enabled but F5 feature maps unavailable')
        # Channel-invariant spatial energy transfer (no teacher/student adapters).
        sf = sf.float().square().mean(1, keepdim=True)
        tf = tf.detach().float().square().mean(1, keepdim=True)
        tf = F.interpolate(tf, size=sf.shape[-2:], mode='bilinear', align_corners=False)
        sf = sf.flatten(1)
        tf = tf.flatten(1)
        sf = F.normalize(sf, p=1, dim=1)
        tf = F.normalize(tf, p=1, dim=1)
        return self.detector_feature_weight * (sf - tf).square().sum(-1).mean()

    def forward(self, outputs, targets, teacher_detect_outputs=None,
                teacher_vfm_flip_features=None, **kwargs):
        # Per-step statistics are not losses and must not leak across iterations.
        self.racofd_diagnostics = {name: 0.0 for name in _RACOFD_DIAG_KEYS}
        # RT-DETRv4 original criterion, unchanged (including DSI/GAM and auxiliary losses).
        losses = super().forward(outputs, targets, **kwargs)
        if not self.training:
            return losses
        zero = outputs['pred_boxes'].sum() * 0
        if teacher_detect_outputs is None:
            if self.use_detector_kd:
                raise RuntimeError('RaCoFD detection KD enabled but teacher_detect_model is None')
            cls_reliable = [torch.zeros(len(t['labels']), device=zero.device) for t in targets]
            box_reliable = [torch.zeros(len(t['labels']), device=zero.device) for t in targets]
            losses.update({'loss_racofd_cls': zero, 'loss_racofd_box': zero})
        else:
            kd, cls_reliable, box_reliable = self._detection_kd(outputs, teacher_detect_outputs, targets)
            losses.update(kd)
        if self.use_instance_vfm:
            losses['loss_racofd_vfm_inst'] = self._instance_vfm(
                outputs, targets, teacher_vfm_flip_features,
                cls_reliable, box_reliable, losses)
        if self.use_detector_feature:
            losses['loss_racofd_xfeat'] = self._detector_feature_kd(outputs, teacher_detect_outputs)
        # Gradually introduce *new* RaCoFD supervision, keeping GT and original
        # DINO/DSI losses unchanged. The engine passes epoch/step/epoch_step.
        scale = 1.0
        if self.new_kd_warmup_epochs > 0:
            progress = float(kwargs.get('epoch', 0)) + (float(kwargs.get('step', 0)) + 1) / max(float(kwargs.get('epoch_step', 1)), 1.0)
            scale = min(1.0, max(0.0, progress / self.new_kd_warmup_epochs))
        self.racofd_diagnostics['racofd_new_kd_scale'] = scale
        for key in tuple(losses):
            if key.startswith('loss_racofd_'):
                losses[key] = losses[key] * scale
        return losses
