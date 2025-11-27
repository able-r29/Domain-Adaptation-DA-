import argparse
import datetime
import os
import shutil
import sys



def copy_file(out) -> None:
    if not os.path.exists(out) or os.path.isfile(out):
        os.makedirs(out)
    else:
        ans = input(out + ' is already exist. continue? ')
        if ans not in ['y', 'yes', 'Y', 'Yes']:
            print('training canceled')
            exit(0)

    def _copy(current='./', dst=os.path.join(out, 'code')):
        if not os.path.exists(dst) or not os.path.isdir(dst):
            os.mkdir(dst)

        items = os.listdir(current)
        for i in items:
            if i in ['.git', '__pycache__']:
                continue
            o = dst + '/' + i
            i = current + '/' + i
            if os.path.isdir(i):
                _copy(i, o)
            elif i.split('.')[-1] in ['py', 'json', 'npz', 'pth', 'txt', 'sh']:
                shutil.copy2(i, o)
    _copy()



def make_script(out, config):
    code_dir = os.path.join(out, 'code')
    command = [
        '#!/bin/bash',
        'ABS_DIR=$(cd $(dirname $0); pwd)'
        'cd $ABS_DIR',
        f'cd {code_dir}',
        f'python3 trainer.py  {config} $@',
        ''
    ]
    command = '\n'.join(command)

    with open('run.sh', 'w') as f:
        f.write(command)


def append_record(date, name):
    path = '../results/record.csv'

    txt  = ','.join([date, name])

    with open(path, 'a') as f:
        print(txt, file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', required=False, default='./results')
    parser.add_argument('--post', '-p', required=False, default=None)
    args = parser.parse_args()

    # config = sys.argv if len(sys.argv) > 1 else input('args : ')

    # model = models.models.format_params()
    # ds    = datasets.dataset_triplet.format_params()
    # train = trainer_triplet.format_params()
    # name  = '__'.join([model, ds, train])

    date = datetime.datetime.today() \
           .astimezone(datetime.timezone(datetime.timedelta(hours=9))) \
           .strftime('%Y_%m_%d__%H_%M_%S')
    if args.post:
        date = f'{date}__{args.post}'
    out = os.path.join(args.root, date)
    copy_file(out)
    print(f'cd {out}/code')


if __name__ == "__main__":
    main()
