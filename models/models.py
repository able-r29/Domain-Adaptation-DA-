# from . import resnet18_mod, resnet18_wd, resnet18_meta, resnet18_cin, \
#               resnet18_mat, resnet18_mse, resnet18_mce, resnet18_mbs, \
#               resnet18_mml, resnet18_base, resnet18_emt, resnet18_trp, \
#               resnet18_mtp, resnet18_amt, resnet18_wtp, resnet18_wtp_weight, \
                # resnet18_wtp_quality_disease, ViT, resnet18_wtp_pred, \
                # resnet18_wtp_dropout, resnet18_loss_weight, resnet18_wtp_meta_wtp, \
                # resnet18_wtp_level1, resnet18_wtp_loss_weight, resnet18_mtp_dropout, resnet18_wtp

from . import resnet18_mod, resnet18_base, resnet18_trp, resnet18_meta, resnet18_mtp

module_map = {
                'resnet18-mod' : resnet18_mod,
                # 'resnet18-wd'  : resnet18_wd,
                'resnet18-meta': resnet18_meta,
                # 'resnet18-cin' : resnet18_cin,
                # 'resnet18-mat' : resnet18_mat,
                # 'resnet18-mse' : resnet18_mse,
                # 'resnet18-mce' : resnet18_mce,
                # 'resnet18-mbs' : resnet18_mbs,
                # 'resnet18-mml' : resnet18_mml,
                'resnet18-base': resnet18_base,
                # 'resnet18-emt' : resnet18_emt,
                'resnet18-trp' : resnet18_trp,
                'resnet18-mtp' : resnet18_mtp,
                # 'resnet18-amt' : resnet18_amt,
                #'resnet18-wtp' : resnet18_wtp,
                # 'resnet18-wtp_weight' : resnet18_wtp_weight,
                # 'resnet18_wtp_qual_dis' : resnet18_wtp_quality_disease,
                # 'ViT' : ViT,
                # 'resnet18_wtp_pred' : resnet18_wtp_pred,
                # 'resnet18_wtp_dropout' : resnet18_wtp_dropout,
                # 'resnet18_loss_weight' : resnet18_loss_weight,
                # 'resnet18-wtp_meta_wtp' : resnet18_wtp_meta_wtp,
                # 'resnet18-wtp_level1' : resnet18_wtp_level1,
                # 'resnet18-wtp_loss_weight' : resnet18_wtp_loss_weight,
                # 'resnet18-mtp_dropout' : resnet18_mtp_dropout
             }


def get_model(name, **karg):
    model = module_map[name].Model(**karg)
    return model


def get_process(name, **karg):
    module      = module_map[name]
    preprocess  = module.get_preprocess (**karg)
    postprocess = module.get_postprocess(**karg)
    loss_func   = module.get_loss_func  (**karg)
    metrics     = module.get_metrics    (**karg)

    return preprocess, postprocess, loss_func, metrics


def get_metrices(name, **karg):
    names = module_map[name].metrices(**karg)
    return names
