"""
DANN Training Engine - 学習ロジック分離版
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import utils


def create_train_step(classifier, domain_discriminator, optimizer, scheduler, 
                     iter_target, device, config, loader_src):
    """DANN学習ステップ"""
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    max_epochs = config['train']['epoch']
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    def train_step(engine, batch):
        classifier.train()
        domain_discriminator.train()
        optimizer.zero_grad()
        
        try:
            # データ準備
            x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=False)
            target_batch = next(iter_target)
            x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=False)
            
            batch_size = min(x_s.shape[0], x_t.shape[0])
            half_size = batch_size // 2
            
            if half_size < 8:
                raise ValueError(f"Batch size too small: {batch_size}")
            
            # 混合バッチ作成
            mixed_x = torch.cat([x_s[:half_size], x_t[:half_size]], dim=0)
            mixed_y_source = y_s[:half_size]
            domain_labels = torch.cat([
                torch.zeros(half_size, 1),
                torch.ones(half_size, 1)
            ], dim=0).to(device)
            
            # GRL強度調整
            p = float(engine.state.iteration + (engine.state.epoch - 1) * len(loader_src)) / (max_epochs * len(loader_src))
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # フォワード
            mixed_pred, mixed_features = classifier(mixed_x)
            source_pred = mixed_pred[:half_size]
            cls_loss = cls_criterion(source_pred, mixed_y_source)
            domain_pred = domain_discriminator(mixed_features)
            domain_loss = domain_criterion(domain_pred, domain_labels)
            total_loss = cls_loss + domain_loss
            
            # バックワード
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # メトリクス計算
            with torch.no_grad():
                source_acc = (source_pred.argmax(dim=1) == mixed_y_source).float().mean()
                
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    cls_auc = 0.5
                
                domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                
                try:
                    domain_auc = roc_auc_score(
                        domain_labels.cpu().numpy().flatten(),
                        domain_pred.cpu().numpy().flatten()
                    )
                except:
                    domain_auc = 0.5
            
            return {
                "loss": total_loss.item(),
                "cls_loss": cls_loss.item(),
                "domain_loss": domain_loss.item(),
                "cls_acc": source_acc.item(),
                "cls_auc": cls_auc,
                "domain_acc": domain_acc.item(),
                "domain_auc": domain_auc,
                "alpha": alpha,
            }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            raise e
    
    return train_step


def create_evaluation_step(classifier, domain_discriminator, loader_target, device, config):
    """評価ステップ"""
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    def evaluation_step(engine, batch):
        classifier.eval()
        domain_discriminator.eval()
        
        with torch.no_grad():
            try:
                # ソースデータ処理
                x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                source_pred, source_features = classifier(x_s)
                cls_loss = cls_criterion(source_pred, y_s)
                source_cls_acc = (source_pred.argmax(dim=1) == y_s).float().mean()
                
                # AUC計算
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    source_auc = 0.5
                
                # ターゲットデータ取得
                try:
                    target_batch = next(loader_target)
                    x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=True)
                    
                    min_batch_size = min(x_s.shape[0], x_t.shape[0])
                    target_pred, target_features = classifier(x_t[:min_batch_size])
                    
                    mixed_features = torch.cat([source_features[:min_batch_size], target_features], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(min_batch_size, 1),
                        torch.ones(min_batch_size, 1)
                    ], dim=0).to(device)
                    
                    # 評価時alpha計算
                    if hasattr(engine, 'state') and hasattr(engine.state, 'epoch'):
                        current_epoch = engine.state.epoch
                        max_epochs = config['train']['epoch']
                        p = float(current_epoch) / max_epochs
                        eval_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                        utils.set_alpha_safely(domain_discriminator, eval_alpha)
                    
                    domain_pred = domain_discriminator(mixed_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                    
                    try:
                        domain_auc = roc_auc_score(
                            domain_labels.cpu().numpy().flatten(),
                            domain_pred.cpu().numpy().flatten()
                        )
                    except:
                        domain_auc = 0.5
                        
                except Exception:
                    domain_labels = torch.zeros(x_s.shape[0], 1).to(device)
                    domain_pred = domain_discriminator(source_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                    domain_auc = 0.5
                
                total_loss = cls_loss + domain_loss
                
                try:
                    cls_macro_sensitivity = utils.macro_sensitivity(
                        source_pred.cpu().numpy(), y_s.cpu().numpy(), source_pred.shape[1]
                    )
                except:
                    cls_macro_sensitivity = 0.0
                
                return {
                    "cls_loss": cls_loss.item(),
                    "cls_accuracy": source_cls_acc.item(),
                    "cls_auc": source_auc,
                    "cls_macro_sensitivity": cls_macro_sensitivity,
                    "domain_loss": domain_loss.item(),
                    "domain_accuracy": domain_acc.item(),
                    "domain_auc": domain_auc,
                    "total_loss": total_loss.item(),
                }
                
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                return {
                    "cls_loss": 1.0,
                    "cls_accuracy": 0.0,
                    "cls_auc": 0.5,
                    "cls_macro_sensitivity": 0.0,
                    "domain_loss": 1.0,
                    "domain_accuracy": 0.5,
                    "domain_auc": 0.5,
                    "total_loss": 2.0,
                }
    
    return evaluation_step