# TEU: Template Ensemble with Uncertainty Estimation
#
# This module provides uncertainty-aware landmark detection by:
# 1. Applying multiple augmentations to the template image
# 2. Generating multiple matching predictions
# 3. Using uncertainty estimation to weight and aggregate predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import random


# ============================================================================
# Template Augmentation
# ============================================================================

class TemplateAugmentor:
    """
    Applies diverse augmentations to the template image for ensemble prediction.
    
    The augmentations are designed to:
    1. Simulate appearance variations (brightness, contrast)
    2. Add small geometric variations (scale, rotation)
    3. Maintain anatomical consistency
    """
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        scale_range: Tuple[float, float] = (0.97, 1.03),
        rotation_range: Tuple[float, float] = (-3, 3),  # degrees
        num_augmentations: int = 5,
        seed: Optional[int] = None
    ):
        """
        Args:
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            scale_range: Range for scale factor
            rotation_range: Range for rotation in degrees
            num_augmentations: Number of augmented versions to generate
            seed: Random seed for reproducibility
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.num_augmentations = num_augmentations
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_augmentation_params(self) -> List[Dict]:
        """Generate parameters for each augmentation."""
        params_list = []
        
        # First one is always identity (no augmentation)
        params_list.append({
            'brightness': 1.0,
            'contrast': 1.0,
            'scale': 1.0,
            'rotation': 0.0
        })
        
        # Generate random augmentations
        for _ in range(self.num_augmentations - 1):
            params = {
                'brightness': random.uniform(*self.brightness_range),
                'contrast': random.uniform(*self.contrast_range),
                'scale': random.uniform(*self.scale_range),
                'rotation': random.uniform(*self.rotation_range)
            }
            params_list.append(params)
        
        return params_list
    
    def apply_augmentation(
        self, 
        image_tensor: torch.Tensor, 
        params: Dict,
        landmarks: Optional[List] = None
    ) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Apply augmentation to image tensor.
        
        Args:
            image_tensor: Input image tensor [C, H, W] or [B, C, H, W]
            params: Augmentation parameters
            landmarks: Optional landmark coordinates to transform
            
        Returns:
            Augmented image tensor and transformed landmarks
        """
        # Handle batch dimension
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, C, H, W = image_tensor.shape
        
        # Apply brightness and contrast
        aug_tensor = image_tensor * params['brightness']
        aug_tensor = (aug_tensor - 0.5) * params['contrast'] + 0.5
        aug_tensor = torch.clamp(aug_tensor, 0, 1)
        
        # Apply scale and rotation using affine transform
        if params['scale'] != 1.0 or params['rotation'] != 0.0:
            angle = params['rotation'] * np.pi / 180
            scale = params['scale']
            
            # Rotation matrix
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            theta = torch.tensor([
                [scale * cos_a, -scale * sin_a, 0],
                [scale * sin_a, scale * cos_a, 0]
            ], dtype=image_tensor.dtype, device=image_tensor.device)
            
            theta = theta.unsqueeze(0).expand(B, -1, -1)
            
            grid = F.affine_grid(theta, aug_tensor.size(), align_corners=False)
            aug_tensor = F.grid_sample(aug_tensor, grid, align_corners=False, mode='bilinear')
            
            # Transform landmarks if provided
            if landmarks is not None:
                transformed_landmarks = []
                cx, cy = W / 2, H / 2
                for lm in landmarks:
                    y, x = lm[0], lm[1]
                    # Translate to center
                    x -= cx
                    y -= cy
                    # Rotate and scale
                    x_new = (x * cos_a - y * sin_a) / scale + cx
                    y_new = (x * sin_a + y * cos_a) / scale + cy
                    transformed_landmarks.append([y_new, x_new])
                landmarks = transformed_landmarks
        
        if squeeze_output:
            aug_tensor = aug_tensor.squeeze(0)
        
        return aug_tensor, landmarks
    
    def augment_template(
        self, 
        image_tensor: torch.Tensor,
        landmarks: Optional[List] = None
    ) -> Tuple[List[torch.Tensor], List[Optional[List]]]:
        """
        Generate multiple augmented versions of the template.
        
        Returns:
            List of augmented tensors and corresponding landmarks
        """
        params_list = self.generate_augmentation_params()
        
        augmented_images = []
        augmented_landmarks = []
        
        for params in params_list:
            aug_img, aug_lm = self.apply_augmentation(image_tensor, params, landmarks)
            augmented_images.append(aug_img)
            augmented_landmarks.append(aug_lm if aug_lm is not None else landmarks)
        
        return augmented_images, augmented_landmarks


# ============================================================================
# Uncertainty Estimation
# ============================================================================

class UncertaintyHead(nn.Module):
    """
    Uncertainty estimation head that predicts epistemic uncertainty
    for each landmark prediction.
    
    Uses the variance of predictions across ensemble members
    plus a learned uncertainty estimate.
    """
    def __init__(self, feature_dim: int, num_landmarks: int, hidden_dim: int = 128):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Predict per-landmark uncertainty from local features
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim + 2, hidden_dim),  # +2 for coordinates
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Learnable base uncertainty
        self.base_uncertainty = nn.Parameter(torch.ones(num_landmarks) * 0.1)
    
    def forward(
        self, 
        coords: torch.Tensor,
        features: torch.Tensor,
        ensemble_variance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate uncertainty for each landmark.
        
        Args:
            coords: Predicted coordinates [B, N, 2]
            features: Local feature vectors [B, N, F]
            ensemble_variance: Variance from ensemble [B, N] (optional)
            
        Returns:
            Uncertainty estimates [B, N]
        """
        B, N, _ = coords.shape
        
        # Concatenate features with coordinates
        input_features = torch.cat([features, coords], dim=-1)  # [B, N, F+2]
        
        # Predict learned uncertainty
        learned_unc = self.uncertainty_net(input_features).squeeze(-1)  # [B, N]
        
        # Add base uncertainty
        total_unc = learned_unc + self.base_uncertainty.unsqueeze(0)
        
        # Add ensemble variance if available
        if ensemble_variance is not None:
            total_unc = total_unc + ensemble_variance
        
        return total_unc


class EnsembleAggregator(nn.Module):
    """
    Aggregates predictions from multiple ensemble members using
    uncertainty-weighted averaging.
    """
    def __init__(self, num_landmarks: int, aggregation: str = 'uncertainty_weighted'):
        """
        Args:
            num_landmarks: Number of landmarks
            aggregation: Aggregation method ('mean', 'median', 'uncertainty_weighted')
        """
        super().__init__()
        self.num_landmarks = num_landmarks
        self.aggregation = aggregation
    
    def forward(
        self, 
        predictions: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate ensemble predictions.
        
        Args:
            predictions: Predictions from ensemble [K, B, N, 2] where K is ensemble size
            uncertainties: Per-prediction uncertainties [K, B, N] (optional)
            
        Returns:
            aggregated: Final predictions [B, N, 2]
            total_uncertainty: Aggregated uncertainty [B, N]
            ensemble_variance: Variance across ensemble [B, N]
        """
        K, B, N, _ = predictions.shape
        
        # Compute ensemble variance
        pred_mean = predictions.mean(dim=0)  # [B, N, 2]
        pred_var = predictions.var(dim=0).sum(dim=-1)  # [B, N]
        ensemble_std = pred_var.sqrt()
        
        if self.aggregation == 'mean':
            aggregated = pred_mean
            
        elif self.aggregation == 'median':
            # Use median for robustness
            aggregated = predictions.median(dim=0).values
            
        elif self.aggregation == 'uncertainty_weighted':
            if uncertainties is None:
                # Use variance-based weighting
                weights = 1.0 / (pred_var.unsqueeze(-1) + 1e-6)  # [B, N, 1]
                weights = weights.unsqueeze(0)  # [1, B, N, 1]
            else:
                # Use provided uncertainties
                weights = 1.0 / (uncertainties.unsqueeze(-1) + 1e-6)  # [K, B, N, 1]
            
            weights = weights / weights.sum(dim=0, keepdim=True)  # Normalize
            aggregated = (predictions * weights).sum(dim=0)  # [B, N, 2]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Total uncertainty combines learned uncertainty and ensemble variance
        if uncertainties is not None:
            total_uncertainty = uncertainties.mean(dim=0) + ensemble_std
        else:
            total_uncertainty = ensemble_std
        
        return aggregated, total_uncertainty, ensemble_std


# ============================================================================
# TEU Module (Complete Pipeline)
# ============================================================================

class TEUModule(nn.Module):
    """
    Template Ensemble with Uncertainty (TEU) Module.
    
    Combines template augmentation, ensemble prediction,
    and uncertainty-aware aggregation for robust landmark detection.
    """
    def __init__(
        self,
        num_landmarks: int = 19,
        feature_dim: int = 256,
        num_augmentations: int = 5,
        aggregation: str = 'uncertainty_weighted',
        use_uncertainty_head: bool = True
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_augmentations = num_augmentations
        
        self.augmentor = TemplateAugmentor(
            num_augmentations=num_augmentations
        )
        
        self.aggregator = EnsembleAggregator(
            num_landmarks=num_landmarks,
            aggregation=aggregation
        )
        
        self.use_uncertainty_head = use_uncertainty_head
        if use_uncertainty_head:
            self.uncertainty_head = UncertaintyHead(
                feature_dim=feature_dim,
                num_landmarks=num_landmarks
            )
    
    def get_ensemble_predictions(
        self,
        extractor,
        model_post,
        template_path: str,
        template_landmarks: List,
        query_batch: torch.Tensor,
        device: str,
        load_size: int = 224,
        layer: int = 9,
        facet: str = 'key',
        bin: bool = True,
        original_size: List = [2400, 1935],
        topk: int = 5,
        use_mlmf: bool = False,
        mlmf_config: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate ensemble predictions using augmented templates.
        
        Returns:
            predictions: [K, B, N, 2]
            scores: [K, B, N]
            features: [K, B, N, F]
        """
        # Load and preprocess template
        from PIL import Image
        template_pil = Image.open(template_path).convert('RGB')
        template_tensor = transforms.ToTensor()(template_pil)
        
        # Generate augmented templates
        aug_templates, aug_landmarks = self.augmentor.augment_template(
            template_tensor, template_landmarks
        )
        
        all_predictions = []
        all_scores = []
        all_features = []
        
        for k, (aug_template, aug_lm) in enumerate(zip(aug_templates, aug_landmarks)):
            # Preprocess augmented template
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            
            if load_size is not None:
                aug_template = F.interpolate(
                    aug_template.unsqueeze(0), 
                    size=load_size, 
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Normalize
            for c in range(3):
                aug_template[c] = (aug_template[c] - mean[c]) / std[c]
            
            aug_template = aug_template.unsqueeze(0).to(device)
            
            # Extract features
            if use_mlmf and mlmf_config is not None:
                desc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    aug_template, 
                    layers=mlmf_config['layers'],
                    facets=mlmf_config['facets'],
                    bin=bin
                )
                num_patches = extractor.num_patches
                desc_post = model_post(desc_list, num_patches)
            else:
                desc = extractor.extract_descriptors(aug_template, layer, facet, bin)
                num_patches = extractor.num_patches
                desc_post = model_post(desc, num_patches)
            
            # Get template landmark features
            lab_features = []
            for i in range(len(aug_lm)):
                lab_y = int(aug_lm[i][0])
                lab_x = int(aug_lm[i][1])
                size_y, size_x = desc_post.shape[-2:]
                lab_y = int(lab_y / original_size[0] * size_y)
                lab_x = int(lab_x / original_size[1] * size_x)
                lab_y = max(0, min(lab_y, size_y - 1))
                lab_x = max(0, min(lab_x, size_x - 1))
                
                lab_feat = desc_post[0, :, lab_y, lab_x]
                lab_features.append(lab_feat)
            
            # Match to query
            predictions, scores, features = self._match_query(
                extractor, model_post, query_batch, lab_features,
                device, load_size, layer, facet, bin, original_size, topk,
                use_mlmf, mlmf_config
            )
            
            all_predictions.append(predictions)
            all_scores.append(scores)
            all_features.append(features)
        
        predictions = torch.stack(all_predictions, dim=0)  # [K, B, N, 2]
        scores = torch.stack(all_scores, dim=0)  # [K, B, N]
        features = torch.stack(all_features, dim=0)  # [K, B, N, F]
        
        return predictions, scores, features
    
    def _match_query(
        self,
        extractor,
        model_post,
        query_batch,
        template_features,
        device,
        load_size,
        layer,
        facet,
        bin,
        original_size,
        topk,
        use_mlmf,
        mlmf_config
    ):
        """Match query images to template features."""
        B = query_batch.shape[0]
        N = len(template_features)
        
        # Extract query features
        if use_mlmf and mlmf_config is not None:
            desc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                query_batch.to(device),
                layers=mlmf_config['layers'],
                facets=mlmf_config['facets'],
                bin=bin
            )
            num_patches = extractor.num_patches
            query_desc = model_post(desc_list, num_patches)
        else:
            desc = extractor.extract_descriptors(query_batch.to(device), layer, facet, bin)
            num_patches = extractor.num_patches
            query_desc = model_post(desc, num_patches)
        
        predictions = []
        scores = []
        features = []
        
        for b in range(B):
            pred_coords = []
            pred_scores = []
            local_feats = []
            
            for i, lab_feat in enumerate(template_features):
                lab_feat_expanded = lab_feat.unsqueeze(1).unsqueeze(2)
                similarities = F.cosine_similarity(
                    lab_feat_expanded, query_desc[b], dim=0
                )
                
                h, w = similarities.shape
                sim_flat = similarities.reshape(-1)
                sim_k, nn_k = torch.topk(sim_flat, k=topk, largest=True)
                
                best_idx = nn_k[0].item()
                best_score = sim_k[0].item()
                
                y_pred = best_idx // w
                x_pred = best_idx % w
                
                local_feat = query_desc[b, :, y_pred, x_pred]
                
                size_y, size_x = query_desc.shape[-2:]
                y_orig = y_pred / size_y * original_size[0]
                x_orig = x_pred / size_x * original_size[1]
                
                pred_coords.append([y_orig, x_orig])
                pred_scores.append(best_score)
                local_feats.append(local_feat)
            
            predictions.append(torch.tensor(pred_coords, device=device))
            scores.append(torch.tensor(pred_scores, device=device))
            features.append(torch.stack(local_feats, dim=0))
        
        predictions = torch.stack(predictions, dim=0)  # [B, N, 2]
        scores = torch.stack(scores, dim=0)  # [B, N]
        features = torch.stack(features, dim=0)  # [B, N, F]
        
        return predictions, scores, features
    
    def forward(
        self,
        predictions: torch.Tensor,
        scores: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate ensemble predictions with uncertainty estimation.
        
        Args:
            predictions: Ensemble predictions [K, B, N, 2]
            scores: Similarity scores [K, B, N]
            features: Local features [K, B, N, F]
            
        Returns:
            final_coords: Aggregated coordinates [B, N, 2]
            uncertainties: Uncertainty estimates [B, N]
        """
        K, B, N, _ = predictions.shape
        
        # Estimate uncertainty for each ensemble member
        if self.use_uncertainty_head:
            uncertainties = []
            for k in range(K):
                unc = self.uncertainty_head(predictions[k], features[k])
                uncertainties.append(unc)
            uncertainties = torch.stack(uncertainties, dim=0)  # [K, B, N]
        else:
            uncertainties = None
        
        # Aggregate predictions
        final_coords, total_unc, ensemble_var = self.aggregator(
            predictions, uncertainties
        )
        
        return final_coords, total_unc


if __name__ == "__main__":
    # Test TEU module
    print("Testing TEU Module...")
    
    # Test augmentor
    augmentor = TemplateAugmentor(num_augmentations=5)
    dummy_image = torch.rand(3, 224, 224)
    dummy_landmarks = [[100, 100], [150, 80], [200, 120]]
    
    aug_images, aug_landmarks = augmentor.augment_template(dummy_image, dummy_landmarks)
    print(f"Generated {len(aug_images)} augmented templates")
    
    # Test uncertainty head
    unc_head = UncertaintyHead(feature_dim=256, num_landmarks=19)
    coords = torch.rand(2, 19, 2)
    features = torch.rand(2, 19, 256)
    uncertainty = unc_head(coords, features)
    print(f"Uncertainty shape: {uncertainty.shape}")
    
    # Test aggregator
    aggregator = EnsembleAggregator(num_landmarks=19, aggregation='uncertainty_weighted')
    ensemble_preds = torch.rand(5, 2, 19, 2)
    ensemble_uncs = torch.rand(5, 2, 19)
    final, total_unc, var = aggregator(ensemble_preds, ensemble_uncs)
    print(f"Aggregated shape: {final.shape}")
    print(f"Total uncertainty shape: {total_unc.shape}")
    
    # Test full TEU module
    teu = TEUModule(num_landmarks=19, feature_dim=256, num_augmentations=5)
    final_coords, uncertainties = teu(ensemble_preds, torch.rand(5, 2, 19), torch.rand(5, 2, 19, 256))
    print(f"Final coords shape: {final_coords.shape}")
    print(f"Uncertainties shape: {uncertainties.shape}")
    
    print("\nTEU Module test passed!")
