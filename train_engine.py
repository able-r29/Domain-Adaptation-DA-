"""
DANN Training Engine - 正しいDANN実装版
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import utils


def create_train_step(classifier, domain_discriminator, optimizer, scheduler, 
                     iter_target_train, device, config, loader_src, start_epoch=1):
    """DANN学習ステップ（修正版）"""
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    max_epochs = config['train']['epoch']
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    epoch_offset = start_epoch - 1
    
    def train_step(engine, batch):
        classifier.train()
        domain_discriminator.train()
        optimizer.zero_grad()
        
        try:
            # ★ 1. ソースデータ処理（train）
            x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=False)
            
            # ★ 2. ターゲットデータ処理（train）
            target_batch = next(iter_target_train)  # ← trainイテレータから取得
            x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=False)
            
            # バッチサイズ調整
            batch_size_s = x_s.shape[0]
            batch_size_t = x_t.shape[0]
            min_batch_size = min(batch_size_s, batch_size_t)
            
            if min_batch_size < 8:
                raise ValueError(f"Batch size too small: source={batch_size_s}, target={batch_size_t}")
            
            # バッチサイズを揃える
            x_s = x_s[:min_batch_size]
            y_s = y_s[:min_batch_size]
            x_t = x_t[:min_batch_size]
            
            # ★ 3. Alpha計算（GRL強度）
            actual_epoch = engine.state.epoch + epoch_offset
            current_iteration = engine.state.iteration + (actual_epoch - 1) * len(loader_src)
            total_iterations = max_epochs * len(loader_src)
            p = float(current_iteration) / total_iterations
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # ★ 4. 特徴抽出
            # ソースデータの特徴抽出と分類
            source_pred, source_features = classifier(x_s)
            
            # ターゲットデータの特徴抽出（分類予測も取得するが損失計算には使わない）
            target_pred, target_features = classifier(x_t)
            
            # ★ 5. 分類損失（ソースデータのみ）
            cls_loss = cls_criterion(source_pred, y_s)
            
            # ★ 6. ドメイン識別損失（ソース + ターゲット）
            # 特徴を結合
            mixed_features = torch.cat([source_features, target_features], dim=0)
            
            # ドメインラベル作成（ソース=0, ターゲット=1）
            domain_labels = torch.cat([
                torch.zeros(min_batch_size, 1),  # ソース
                torch.ones(min_batch_size, 1)    # ターゲット
            ], dim=0).to(device)
            
            # ドメイン識別
            domain_pred = domain_discriminator(mixed_features)
            domain_loss = domain_criterion(domain_pred, domain_labels)
            
            # ★ 7. 合計損失
            total_loss = cls_loss + domain_loss
            
            # ★ 8. 逆伝播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # ★ 9. メトリクス計算
            with torch.no_grad():
                # ソース分類精度
                source_acc = (source_pred.argmax(dim=1) == y_s).float().mean()
                
                # ソース分類AUC
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        cls_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        cls_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    cls_auc = 0.5
                
                # ドメイン識別精度
                domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                
                # ドメイン識別AUC
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
                "p": p,
                "actual_epoch": actual_epoch,
                "batch_size": min_batch_size,
            }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            raise e
    
    return train_step


def create_evaluation_step(classifier, domain_discriminator, iter_target_val, 
                          device, config, start_epoch=1):
    """評価ステップ（validationイテレータ版）"""
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    epoch_offset = start_epoch - 1
    max_epochs = config['train']['epoch']
    
    def evaluation_step(engine, batch):
        classifier.eval()
        domain_discriminator.eval()
        
        with torch.no_grad():
            try:
                # ★ 1. ソースデータ処理（validation）
                x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                source_pred, source_features = classifier(x_s)
                cls_loss = cls_criterion(source_pred, y_s)
                source_cls_acc = (source_pred.argmax(dim=1) == y_s).float().mean()
                
                # ソースAUC計算
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    source_auc = 0.5
                
                # ★ 2. Alpha計算
                if hasattr(engine, 'state') and hasattr(engine.state, 'epoch'):
                    actual_epoch = engine.state.epoch + epoch_offset
                    p = float(actual_epoch) / max_epochs
                    eval_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                    utils.set_alpha_safely(domain_discriminator, eval_alpha)
                
                # ★ 3. ターゲットデータ処理（validation）
                target_cls_acc = None
                target_auc = None
                
                try:
                    target_batch = next(iter_target_val)  # ← validationイテレータから取得
                    x_t, y_t = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=True)
                    
                    min_batch_size = min(x_s.shape[0], x_t.shape[0])
                    target_pred, target_features = classifier(x_t[:min_batch_size])
                    
                    # ★ ターゲットの分類精度（研究用・validationのみ）
                    if y_t is not None and len(y_t) > 0:
                        target_cls_acc = (target_pred.argmax(dim=1) == y_t[:min_batch_size]).float().mean()
                        
                        # ターゲットAUC
                        try:
                            if target_pred.shape[1] == 2:
                                target_pred_prob = torch.softmax(target_pred, dim=1)[:, 1].cpu().numpy()
                                target_auc = roc_auc_score(y_t[:min_batch_size].cpu().numpy(), target_pred_prob)
                            else:
                                target_pred_prob = torch.softmax(target_pred, dim=1).cpu().numpy()
                                target_auc = roc_auc_score(y_t[:min_batch_size].cpu().numpy(), target_pred_prob, multi_class='ovr')
                        except:
                            target_auc = 0.5
                    
                    # ★ 4. ドメイン識別損失（validation）
                    mixed_features = torch.cat([source_features[:min_batch_size], target_features], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(min_batch_size, 1),
                        torch.ones(min_batch_size, 1)
                    ], dim=0).to(device)
                    
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
                        
                except Exception as e:
                    print(f"⚠️ Target validation evaluation failed: {e}")
                    domain_labels = torch.zeros(x_s.shape[0], 1).to(device)
                    domain_pred = domain_discriminator(source_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                    domain_auc = 0.5
                
                total_loss = cls_loss + domain_loss
                
                # Macro Sensitivity
                try:
                    cls_macro_sensitivity = utils.macro_sensitivity(
                        source_pred.cpu().numpy(), y_s.cpu().numpy(), source_pred.shape[1]
                    )
                except:
                    cls_macro_sensitivity = 0.0
                
                result = {
                    "cls_loss": cls_loss.item(),
                    "cls_accuracy": source_cls_acc.item(),
                    "cls_auc": source_auc,
                    "cls_macro_sensitivity": cls_macro_sensitivity,
                    "domain_loss": domain_loss.item(),
                    "domain_accuracy": domain_acc.item(),
                    "domain_auc": domain_auc,
                    "total_loss": total_loss.item(),
                }
                
                # ★ ターゲット性能を追加（validationのみ）
                if target_cls_acc is not None:
                    result["target_cls_accuracy"] = target_cls_acc.item()
                if target_auc is not None:
                    result["target_cls_auc"] = target_auc
                
                return result
                
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                import traceback
                traceback.print_exc()
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