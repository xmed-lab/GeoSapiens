# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmpose.registry import MODELS

@MODELS.register_module()
class GeometricalKeypointMSELoss(nn.Module):
    """MSE loss for heatmaps with geometrical constraints."""

    def __init__(
        self,
        use_target_weight: bool = False,
        skip_empty_channel: bool = False,
        loss_weight: float = 1.0,
        geometric_weight: float = 1e-5,
    ):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight
        self.geometric_weight = geometric_weight

    def heatmap_to_coord(self, heatmap, temperature=0.1):
        """Convert heatmap to coordinate using soft-argmax.

        Args:
            heatmap (Tensor): Heatmap with shape [B, K, H, W]
            temperature (float): Temperature parameter for softmax. Lower values make the distribution sharper.

        Returns:
            Tensor: Coordinates with shape [B, K, 2] (x, y)
        """
        B, K, H, W = heatmap.shape

        # Flatten the spatial dimensions
        heatmap_flat = heatmap.reshape(B, K, -1)  # Shape: [B, K, H*W]

        # Apply softmax to convert heatmap to a probability distribution
        prob = F.softmax(heatmap_flat / temperature, dim=2)  # Shape: [B, K, H*W]

        # Create coordinate grid
        pos_x, pos_y = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=heatmap.device),
            torch.arange(H, dtype=torch.float32, device=heatmap.device),
            indexing="xy",
        )
        pos_x = pos_x.reshape(-1)  # Flatten, shape: [H*W]
        pos_y = pos_y.reshape(-1)  # Flatten, shape: [H*W]

        # Make sure coordinates are part of computation graph
        pos_x = pos_x.expand(B, K, -1)  # Shape: [B, K, H*W]
        pos_y = pos_y.expand(B, K, -1)  # Shape: [B, K, H*W]

        # Compute weighted average coordinates
        x_coord = torch.sum(prob * pos_x, dim=2)  # Shape: [B, K]
        y_coord = torch.sum(prob * pos_y, dim=2)  # Shape: [B, K]

        # Stack x,y coordinates
        coords = torch.stack([x_coord, y_coord], dim=2)  # Shape: [B, K, 2]
        # print(f"coords: {coords}")

        return coords


    def get_unit_vector(self, line_params):
        """Calculate unit vector from line parameters."""
        # use existing tensor to create vector, keep gradient flow
        vector = torch.stack([torch.ones_like(line_params[0]), line_params[0]])
        norm = torch.norm(vector)
        return vector / (norm + 1e-6)

    def compute_geometric_loss(self, coords):
        """Calculate geometric constraint loss in a vectorized way.

        Args:
            coords (Tensor): Coordinates with shape [B, K, 2]

        Returns:
            Tensor: Geometric constraint loss
        """


        points_2_5 = coords[:, [2, 5]]  # [B, 2, 2]
        points_2_6_7 = coords[:, [2, 6, 7]]  # [B, 3, 2]
        points_8_11 = coords[:, [8, 9, 10, 11]]  # [B, 4, 2]
        points_12_15 = coords[:, [12, 13, 14, 15]]  # [B, 4, 2]

        # Fit lines for all batches
        line_2_5 = self.fit_line_batch(points_2_5)  # [B, 2]
        line_2_6_7 = self.fit_line_batch(points_2_6_7)  # [B, 2]
        line_8_11 = self.fit_line_batch(points_8_11)  # [B, 2]
        line_12_15 = self.fit_line_batch(points_12_15)  # [B, 2]

        # Calculate unit vectors for all lines
        v_2_5 = self.get_unit_vector_batch(line_2_5)  # [B, 2]
        v_2_6_7 = self.get_unit_vector_batch(line_2_6_7)  # [B, 2]
        v_8_11 = self.get_unit_vector_batch(line_8_11)  # [B, 2]
        v_12_15 = self.get_unit_vector_batch(line_12_15)  # [B, 2]

        # Stack vectors for dot product calculation
        vectors = torch.stack([v_2_6_7, v_8_11, v_12_15], dim=1)  # [B, 3, 2]

        # Calculate orthogonality loss with v_2_5
        ortho_loss = torch.abs(torch.bmm(vectors, v_2_5.unsqueeze(2))).mean(
            dim=(1, 2)
        )  # [B]

        # Calculate relationships between other vectors
        dot_products = torch.bmm(vectors, vectors.transpose(1, 2))  # [B, 3, 3]

        # Get upper triangle indices for each batch
        triu_indices = torch.triu_indices(3, 3, offset=1)
        parallel_loss = (
            1 - torch.abs(dot_products[:, triu_indices[0], triu_indices[1]])
        ).mean(
            dim=1
        )  # [B]
        loss = (ortho_loss + parallel_loss).mean()

        return loss

    def fit_line_batch(self, points):
        """Fit lines for a batch of points using least squares method.

        Args:
            points (Tensor): Points with shape [B, N, 2] where N is number of points per line

        Returns:
            Tensor: Line parameters [slope, intercept] for each batch [B, 2]
        """
        x = points[..., 0]  # [B, N]
        y = points[..., 1]  # [B, N]

        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1]
        y_mean = y.mean(dim=1, keepdim=True)  # [B, 1]

        # Compute slope
        numerator = ((x - x_mean) * (y - y_mean)).sum(dim=1)  # [B]
        denominator = ((x - x_mean) ** 2).sum(dim=1)  # [B]

        slope = numerator / (denominator + 1e-6)  # [B]

        # Compute intercept
        intercept = y_mean.squeeze(1) - slope * x_mean.squeeze(1)  # [B]

        return torch.stack([slope, intercept], dim=1)  # [B, 2]

    def get_unit_vector_batch(self, line_params):
        """Calculate unit vectors from line parameters in batch.

        Args:
            line_params (Tensor): Line parameters with shape [B, 2]

        Returns:
            Tensor: Unit vectors with shape [B, 2]
        """
        vector = torch.stack(
            [torch.ones_like(line_params[:, 0]), line_params[:, 0]], dim=1
        )  # [B, 2]

        norm = torch.norm(vector, dim=1, keepdim=True)  # [B, 1]
        return vector / (norm + 1e-6)  # [B, 2]


    

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        target_weights: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function of loss."""
        # Convert heatmap to coordinates
        coords = self.heatmap_to_coord(output)
        target_coords = target.reshape(target.shape[0], target.shape[1], -1).argmax(
            dim=2
        )
        target_coords = torch.stack(
            [target_coords % target.shape[3], target_coords // target.shape[3]], dim=2
        ).float()
    
        # Calculate geometric loss (first part)
        geometric_loss = self.compute_geometric_loss(coords)

        # Calculate MSE loss
        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            mse_loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction="none")
            mse_loss = (_loss * _mask).mean()
        
        # Combine all losses - 确保包含所有损失项
        total_loss = (
            mse_loss
            + self.geometric_weight * geometric_loss
        )

        return total_loss

    def _get_mask(
        self, target: Tensor, target_weights: Optional[Tensor], mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1 for d_m, d_t in zip(mask.shape, target.shape)
            ), (
                f"mask and target have mismatched shapes {mask.shape} v.s."
                f"{target.shape}"
            )

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (
                target_weights.ndim in (2, 4)
                and target_weights.shape == target.shape[: target_weights.ndim]
            ), (
                "target_weights and target have mismatched shapes "
                f"{target_weights.shape} v.s. {target.shape}"
            )

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1,) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask

@MODELS.register_module()
class KeypointMSELoss(nn.Module):
    """MSE loss for heatmaps.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.skip_empty_channel = skip_empty_channel
        self.loss_weight = loss_weight

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """

        _mask = self._get_mask(target, target_weights, mask)
        if _mask is None:
            loss = F.mse_loss(output, target)
        else:
            _loss = F.mse_loss(output, target, reduction='none')
            loss = (_loss * _mask).mean()

        return loss * self.loss_weight

    def _get_mask(self, target: Tensor, target_weights: Optional[Tensor],
                  mask: Optional[Tensor]) -> Optional[Tensor]:
        """Generate the heatmap mask w.r.t. the given mask, target weight and
        `skip_empty_channel` setting.

        Returns:
            Tensor: The mask in shape (B, K, *) or ``None`` if no mask is
            needed.
        """
        # Given spatial mask
        if mask is not None:
            # check mask has matching type with target
            assert (mask.ndim == target.ndim and all(
                d_m == d_t or d_m == 1
                for d_m, d_t in zip(mask.shape, target.shape))), (
                    f'mask and target have mismatched shapes {mask.shape} v.s.'
                    f'{target.shape}')

        # Mask by target weights (keypoint-wise mask)
        if target_weights is not None:
            # check target weight has matching shape with target
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            _mask = target_weights.view(target_weights.shape +
                                        (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        # Mask by ``skip_empty_channel``
        if self.skip_empty_channel:
            _mask = (target != 0).flatten(2).any(dim=2)
            ndim_pad = target.ndim - _mask.ndim
            _mask = _mask.view(_mask.shape + (1, ) * ndim_pad)

            if mask is None:
                mask = _mask
            else:
                mask = mask * _mask

        return mask


@MODELS.register_module()
class CombinedTargetMSELoss(nn.Module):
    """MSE loss for combined target.

    CombinedTarget: The combination of classification target
    (response map) and regression target (offset map).
    Paper ref: Huang et al. The Devil is in the Details: Delving into
    Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 loss_weight: float = 1.):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_channels: C
            - heatmaps height: H
            - heatmaps weight: W
            - num_keypoints: K
            Here, C = 3 * K

        Args:
            output (Tensor): The output feature maps with shape [B, C, H, W].
            target (Tensor): The target feature maps with shape [B, C, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        batch_size = output.size(0)
        num_channels = output.size(1)
        heatmaps_pred = output.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_channels, -1)).split(1, 1)
        loss = 0.
        num_joints = num_channels // 3
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx * 3].squeeze()
            heatmap_gt = heatmaps_gt[idx * 3].squeeze()
            offset_x_pred = heatmaps_pred[idx * 3 + 1].squeeze()
            offset_x_gt = heatmaps_gt[idx * 3 + 1].squeeze()
            offset_y_pred = heatmaps_pred[idx * 3 + 2].squeeze()
            offset_y_gt = heatmaps_gt[idx * 3 + 2].squeeze()
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None]
                heatmap_pred = heatmap_pred * target_weight
                heatmap_gt = heatmap_gt * target_weight
            # classification loss
            loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
            # regression loss
            loss += 0.5 * self.criterion(heatmap_gt * offset_x_pred,
                                         heatmap_gt * offset_x_gt)
            loss += 0.5 * self.criterion(heatmap_gt * offset_y_pred,
                                         heatmap_gt * offset_y_gt)
        return loss / num_joints * self.loss_weight


@MODELS.register_module()
class KeypointOHKMMSELoss(nn.Module):
    """MSE loss with online hard keypoint mining.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        topk (int): Only top k joint losses are kept. Defaults to 8
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 use_target_weight: bool = False,
                 topk: int = 8,
                 loss_weight: float = 1.):
        super().__init__()
        assert topk > 0
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.loss_weight = loss_weight

    def _ohkm(self, losses: Tensor) -> Tensor:
        """Online hard keypoint mining.

        Note:
            - batch_size: B
            - num_keypoints: K

        Args:
            loss (Tensor): The losses with shape [B, K]

        Returns:
            Tensor: The calculated loss.
        """
        ohkm_loss = 0.
        B = losses.shape[0]
        for i in range(B):
            sub_loss = losses[i]
            _, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= B
        return ohkm_loss

    def forward(self, output: Tensor, target: Tensor,
                target_weights: Tensor) -> Tensor:
        """Forward function of loss.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W].
            target (Tensor): The target heatmaps with shape [B, K, H, W].
            target_weights (Tensor): The target weights of differet keypoints,
                with shape [B, K].

        Returns:
            Tensor: The calculated loss.
        """
        num_keypoints = output.size(1)
        if num_keypoints < self.topk:
            raise ValueError(f'topk ({self.topk}) should not be '
                             f'larger than num_keypoints ({num_keypoints}).')

        losses = []
        for idx in range(num_keypoints):
            if self.use_target_weight:
                target_weight = target_weights[:, idx, None, None]
                losses.append(
                    self.criterion(output[:, idx] * target_weight,
                                   target[:, idx] * target_weight))
            else:
                losses.append(self.criterion(output[:, idx], target[:, idx]))

        losses = [loss.mean(dim=(1, 2)).unsqueeze(dim=1) for loss in losses]
        losses = torch.cat(losses, dim=1)

        return self._ohkm(losses) * self.loss_weight


@MODELS.register_module()
class AdaptiveWingLoss(nn.Module):
    """Adaptive wing loss. paper ref: 'Adaptive Wing Loss for Robust Face
    Alignment via Heatmap Regression' Wang et al. ICCV'2019.

    Args:
        alpha (float), omega (float), epsilon (float), theta (float)
            are hyper-parameters.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 alpha=2.1,
                 omega=14,
                 epsilon=1,
                 theta=0.5,
                 use_target_weight=False,
                 loss_weight=1.):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)
        self.use_target_weight = use_target_weight
        self.loss_weight = loss_weight

    def criterion(self, pred, target):
        """Criterion of wingloss.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            pred (torch.Tensor[NxKxHxW]): Predicted heatmaps.
            target (torch.Tensor[NxKxHxW]): Target heatmaps.
        """
        H, W = pred.shape[2:4]
        delta = (target - pred).abs()

        A = self.omega * (
            1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        ) * (self.alpha - target) * (torch.pow(
            self.theta / self.epsilon,
            self.alpha - target - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target))

        losses = torch.where(
            delta < self.theta,
            self.omega *
            torch.log(1 +
                      torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C)

        return torch.mean(losses)

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None):
        """Forward function.

        Note:
            batch_size: N
            num_keypoints: K

        Args:
            output (torch.Tensor[N, K, H, W]): Output heatmaps.
            target (torch.Tensor[N, K, H, W]): Target heatmaps.
            target_weight (torch.Tensor[N, K]):
                Weights across different joint types.
        """
        if self.use_target_weight:
            assert (target_weights.ndim in (2, 4) and target_weights.shape
                    == target.shape[:target_weights.ndim]), (
                        'target_weights and target have mismatched shapes '
                        f'{target_weights.shape} v.s. {target.shape}')

            ndim_pad = target.ndim - target_weights.ndim
            target_weights = target_weights.view(target_weights.shape +
                                                 (1, ) * ndim_pad)
            loss = self.criterion(output * target_weights,
                                  target * target_weights)
        else:
            loss = self.criterion(output, target)

        return loss * self.loss_weight


@MODELS.register_module()
class FocalHeatmapLoss(KeypointMSELoss):
    """A class for calculating the modified focal loss for heatmap prediction.

    This loss function is exactly the same as the one used in CornerNet. It
    runs faster and costs a little bit more memory.

    `CornerNet: Detecting Objects as Paired Keypoints
    arXiv: <https://arxiv.org/abs/1808.01244>`_.

    Arguments:
        alpha (int): The alpha parameter in the focal loss equation.
        beta (int): The beta parameter in the focal loss equation.
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
            Defaults to ``False``
        skip_empty_channel (bool): If ``True``, heatmap channels with no
            non-zero value (which means no visible ground-truth keypoint
            in the image) will not be used to calculate the loss. Defaults to
            ``False``
        loss_weight (float): Weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 alpha: int = 2,
                 beta: int = 4,
                 use_target_weight: bool = False,
                 skip_empty_channel: bool = False,
                 loss_weight: float = 1.0):
        super(FocalHeatmapLoss, self).__init__(use_target_weight,
                                               skip_empty_channel, loss_weight)
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                output: Tensor,
                target: Tensor,
                target_weights: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """Calculate the modified focal loss for heatmap prediction.

        Note:
            - batch_size: B
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (Tensor): The output heatmaps with shape [B, K, H, W]
            target (Tensor): The target heatmaps with shape [B, K, H, W]
            target_weights (Tensor, optional): The target weights of differet
                keypoints, with shape [B, K] (keypoint-wise) or
                [B, K, H, W] (pixel-wise).
            mask (Tensor, optional): The masks of valid heatmap pixels in
                shape [B, K, H, W] or [B, 1, H, W]. If ``None``, no mask will
                be applied. Defaults to ``None``

        Returns:
            Tensor: The calculated loss.
        """
        _mask = self._get_mask(target, target_weights, mask)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        if _mask is not None:
            pos_inds = pos_inds * _mask
            neg_inds = neg_inds * _mask

        neg_weights = torch.pow(1 - target, self.beta)

        pos_loss = torch.log(output) * torch.pow(1 - output,
                                                 self.alpha) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(
            output, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss.sum()
        else:
            loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss * self.loss_weight
