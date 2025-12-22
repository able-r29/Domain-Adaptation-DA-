"""
Domain Discriminator for DANN (Domain-Adversarial Neural Networks)
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel


class GradientReverseFunction(torch.autograd.Function):
    """勾配反転レイヤーの実装"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReverseLayer(nn.Module):
    """勾配反転レイヤー"""
    def __init__(self, alpha=1.0):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha


class DomainDiscriminator(nn.Module):
    """ドメイン識別器（動的特徴量次元対応）"""
    def __init__(self, initial_feature_dim=512, hidden_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReverseLayer()
        self.initial_feature_dim = initial_feature_dim
        self.hidden_dim = hidden_dim
        self.current_feature_dim = None
        self.classifier = None
        
        print(f"🏗️ DomainDiscriminator initialized with initial_feature_dim={initial_feature_dim}")
    
    def _create_classifier(self, feature_dim, device):
        """特徴量次元に基づいて分類器を動的に作成"""
        if self.current_feature_dim != feature_dim:
            print(f"🔧 Creating domain discriminator layers for feature dim: {feature_dim}")
            
            self.current_feature_dim = feature_dim
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.BatchNorm1d(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid()
            ).to(device)
            
            print(f"✓ Domain discriminator layers created: {feature_dim} -> {self.hidden_dim} -> 1")
    
    def forward(self, x):
        # ★ ここで勾配反転レイヤーを通過（重要）
        x = self.grl(x)  # 順伝播：そのまま通す / 逆伝播：勾配を反転
        
        # 動的に分類器を作成
        current_feature_dim = x.size(1)
        if self.classifier is None or self.current_feature_dim != current_feature_dim:
            self._create_classifier(current_feature_dim, x.device)
        
        # ドメイン識別
        return self.classifier(x)
    
    def set_alpha(self, alpha):
        """GRLのalpha値を設定"""
        self.grl.set_alpha(alpha)  # α=1.0に設定


class DANNClassifier(nn.Module):
    """DANN用分類器（resnet18_mtp対応・修正版）"""
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        super(DANNClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        self.current_feature_dim = None
        self.bottleneck = None
        self.classifier = None
        
        print(f"🏗️ DANNClassifier initialized:")
        print(f"  Backbone: {type(backbone).__name__}")
        print(f"  Bottleneck dim: {bottleneck_dim}")
        print(f"  Num classes: {num_classes}")
        
    def _create_layers(self, feature_dim, device):
        """特徴量次元に基づいてボトルネック層と分類ヘッドを動的に作成"""
        if self.current_feature_dim != feature_dim:
            print(f"🔧 Creating classifier layers for feature dim: {feature_dim}")
            
            self.current_feature_dim = feature_dim
            
            # ボトルネック層を作成
            self.bottleneck = nn.Sequential(
                nn.Linear(feature_dim, self.bottleneck_dim),
                nn.BatchNorm1d(self.bottleneck_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ).to(device)
            
            # 分類ヘッドを作成
            self.classifier = nn.Linear(self.bottleneck_dim, self.num_classes).to(device)
            
            print(f"✓ Classifier layers created: {feature_dim} -> {self.bottleneck_dim} -> {self.num_classes}")
    
    def _prepare_backbone_input(self, x):
        """バックボーン用の入力を準備（resnet18_mtp対応）"""
        # resnet18_mtpは辞書形式を期待: {'anchor': tensor, 'positive': tensor, 'meta_a': tensor, 'meta_p': tensor}
        
        if isinstance(x, dict):
            # 既に辞書形式の場合
            if 'anchor' in x:
                # 完全な辞書が渡された場合はそのまま使用
                return x
            else:
                # anchorキーがない場合は最初のテンソルをanchorとして使用
                tensor_keys = [k for k, v in x.items() if isinstance(v, torch.Tensor)]
                if tensor_keys:
                    anchor_tensor = x[tensor_keys[0]]
                    # ダミーのmeta情報を作成（resnet18_mtpが要求する形式）
                    batch_size = anchor_tensor.size(0)
                    device = anchor_tensor.device
                    
                    return {
                        'anchor': anchor_tensor,
                        'positive': anchor_tensor,  # ダミー：anchorと同じ
                        'meta_a': torch.zeros(batch_size, 1, device=device),  # ダミーメタ情報
                        'meta_p': torch.zeros(batch_size, 1, device=device)   # ダミーメタ情報
                    }
                else:
                    raise ValueError(f"No tensor found in input dict: {list(x.keys())}")
        
        elif isinstance(x, torch.Tensor):
            # 4次元テンソルが直接渡された場合
            batch_size = x.size(0)
            device = x.device
            
            return {
                'anchor': x,
                'positive': x,  # ダミー：anchorと同じ
                'meta_a': torch.zeros(batch_size, 1, device=device),  # ダミーメタ情報
                'meta_p': torch.zeros(batch_size, 1, device=device)   # ダミーメタ情報
            }
        else:
            raise ValueError(f"Unexpected input type: {type(x)}")
        
    def forward(self, x):
        # バックボーン用の入力を準備
        backbone_input = self._prepare_backbone_input(x)
        
        # バックボーンで特徴抽出（resnet18_mtpのfeatureメソッドを使用）
        if hasattr(self.backbone, 'feature'):
            # featureメソッドがある場合（anchorのみの特徴抽出）
            features = self.backbone.feature(backbone_input)
        else:
            # 通常のforwardを使用してanchor特徴量を抽出
            try:
                output = self.backbone(backbone_input)
                if isinstance(output, dict) and 'ya' in output:
                    # resnet18_mtpの場合、predictメソッドを使用してfeatureを取得
                    if hasattr(self.backbone, 'predict'):
                        _, features = self.backbone.predict(backbone_input)
                    else:
                        # フォールバック：yaの前の層の出力を取得
                        features = self.backbone.feature(backbone_input)
                else:
                    features = output
            except Exception as e:
                print(f"⚠️ Backbone forward failed: {e}")
                # 最後の手段：featureメソッドを試行
                features = self.backbone.feature(backbone_input)
        
        # 特徴量を平坦化
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # 動的に層を作成
        current_feature_dim = features.size(1)
        if self.bottleneck is None or self.current_feature_dim != current_feature_dim:
            self._create_layers(current_feature_dim, features.device)
        
        # ボトルネック層を通す
        bottleneck_features = self.bottleneck(features)
        
        # 分類
        logits = self.classifier(bottleneck_features)
        
        return logits, bottleneck_features


def calculate_lambda_p(epoch, max_epochs):
    """学習進行に応じてGRLの強度を調整"""
    import numpy as np
    p = float(epoch) / max_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0