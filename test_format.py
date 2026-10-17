import os
import pandas as pd
import numpy as np

df = pd.read_csv('analysis/thesis_rbase_assessment/aggregated_metrics.csv')

def format_acc(mean, std, is_best=False):
    s = f'{mean:.2f}'
    if std > 0.005:
        s += f' \\pm {std:.2f}'
    if is_best:
        return f'\\mathbf{{{s}}}'
    return f'$'

def format_time(mean, std, is_best=False):
    if mean >= 100:
        s = f'{mean:.1f}'
        if std > 0.05:
            s += f' \\pm {std:.1f}'
    else:
        s = f'{mean:.2f}'
        if std > 0.005:
            s += f' \\pm {std:.2f}'
    if is_best:
        return f'\\mathbf{{{s}}}'
    return f'$'

def format_round(mean, std):
    if std > 0.05:
        return f' \\pm {std:.1f}$'
    return f'$'

# Test format
print('Testing format functions:')
print(format_acc(92.43, 0.16, True))
print(format_time(414.0, 62.8, False))
