import json
import sys

import optuna



def minmax_suggest(trial:optuna.Trial, src:dict):
    name = src['args']['name']
    src['opttype'] = src['opttype'].replace('minmax', '')

    src['args']['name'] = name + '1'
    val1 = get_suggest(trial, src)

    src['args']['name'] = name + '2'
    val2 = get_suggest(trial, src)

    vmin = min(val1, val2)
    vmax = max(val1, val2)
    return vmin, vmax


def get_suggest(trial:optuna.Trial, src:dict):
    if src['opttype'] == 'minmaxuniform':
        return minmax_suggest(trial, src)

    func = {
        'uniform'    :trial.suggest_uniform,
        'loguniform' :trial.suggest_loguniform,
        'categorical':trial.suggest_categorical
    }[src['opttype']]

    val = func(**src['args'])
    return val



def parse_dict(trial:optuna.Trial, src:dict):
    dst = {}
    for k, v in src.items():
        if isinstance(v, dict) and 'opttype' in v:
            v = get_suggest(trial, v)
        dst[k] = parse(trial, v)

    return dst



def parse_list(trial:optuna.Trial, src:list):
    dst = []
    for v in src:
        if isinstance(v, dict) and 'opttype' in v:
            v = get_suggest(trial, v)
        dst.append(parse(trial, v))

    return dst



def parse(trial:optuna.Trial, src):
    if isinstance(src, dict):
        return parse_dict(trial, src)
    if isinstance(src, list) or isinstance(src, tuple):
        return parse_list(trial, src)

    return src



def get_search_log():
    name = sys.argv[1] if len(sys.argv) > 1 else input('study naem: ')
    print(name)

    study = optuna.create_study(study_name=name,
                                storage="mysql+pymysql://root:hogehoge@192.168.1.39:13306/ptlr",
                                load_if_exists=True)
    funcs = [
        optuna.visualization.plot_contour,
        optuna.visualization.plot_edf,
        optuna.visualization.plot_intermediate_values,
        optuna.visualization.plot_optimization_history,
        optuna.visualization.plot_parallel_coordinate,
        optuna.visualization.plot_param_importances,
        # optuna.visualization.plot_pareto_front,
        optuna.visualization.plot_slice
    ]
    for f in funcs:
        fig = f(study)
        fig.show()


if __name__ == '__main__':
    get_search_log()
