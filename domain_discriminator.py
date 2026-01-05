"""
Domain Discriminator for DANN (resnet18_base対応版)
"""

import torch
import torch.nn as nn


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
    """ドメイン識別器（resnet18_base用・256次元特徴量対応）"""
    def __init__(self, feature_dim=256, hidden_dim=1024):
        super(DomainDiscriminator, self).__init__()
        self.grl = GradientReverseLayer()
        
        # 固定された構造（resnet18_base用）
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        print(f"🏗️ DomainDiscriminator initialized:")
        print(f"  Input feature dim: {feature_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Architecture: {feature_dim} -> {hidden_dim} -> {hidden_dim} -> 1")
    
    def forward(self, x):
        # ★ 勾配反転レイヤーを通過（重要）
        x = self.grl(x)
        return self.classifier(x)
    
    def set_alpha(self, alpha):
        """GRLのalpha値を設定"""
        self.grl.set_alpha(alpha)


class DANNClassifier(nn.Module):
    """DANN用分類器（resnet18_base専用・シンプル版）"""
    def __init__(self, backbone, num_classes, bottleneck_dim=256):
        super(DANNClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim
        
        # resnet18_baseは512次元特徴量を出力
        self.bottleneck = nn.Sequential(
            nn.Linear(512, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 分類ヘッド
        self.classifier_head = nn.Linear(bottleneck_dim, num_classes)
        
        print(f"🏗️ DANNClassifier initialized:")
        print(f"  Backbone: resnet18_base (512 features)")
        print(f"  Bottleneck: 512 -> {bottleneck_dim}")
        print(f"  Classifier: {bottleneck_dim} -> {num_classes}")
    
    def forward(self, x):
        """
        Args:
            x: 画像テンソル [batch_size, 3, height, width]
        Returns:
            logits: 分類スコア [batch_size, num_classes]
            bottleneck_features: ボトルネック特徴 [batch_size, bottleneck_dim]
        """
        # バックボーンで特徴抽出（512次元）
        backbone_features = self.backbone.feature(x)
        
        # ボトルネック層を通す（256次元）
        bottleneck_features = self.bottleneck(backbone_features)
        
        # 分類
        logits = self.classifier_head(bottleneck_features)
        
        return logits, bottleneck_features


def calculate_lambda_p(epoch, max_epochs):
    """学習進行に応じてGRLの強度を調整"""
    import numpy as np
    p = float(epoch) / max_epochs
    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0