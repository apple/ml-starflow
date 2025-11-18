#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os


def get_local_rank():
    if os.environ.get('IRISCTL_ROLE'): 
        import irisctl.api as irisctl
        return irisctl.local_rank()
    elif os.environ.get('MASTER_PORT'):
        return int(os.environ['LOCAL_RANK'])
    else:
        return 0


def print(*args, **kwargs):
    if get_local_rank() == 0:
        import builtins
        builtins.print(*args, **kwargs)


def xprint(string):
    import builtins
    local_rank = get_local_rank()
    builtins.print(f'[Local Rank {local_rank}] {string}')


def dividable(x):
    for i in range(int(x ** 0.5), 0, -1):
        if x % i == 0:
            return x // i
    return x
