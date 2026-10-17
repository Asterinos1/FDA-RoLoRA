import glob
import os
import re
import numpy as np
import pandas as pd

def prep(df):
    if 'round' in df.columns:
        df = df.sort_values('round').reset_index(drop=True)
    if 'accuracy' not in df.columns and {'accuracy_m', 'accuracy_mm'}.issubset(df.columns):
        df['accuracy'] = (df['accuracy_m'] + df['accuracy_mm']) / 2.0
    if 'round_epochs' not in df.columns:
        if 'epochs' in df.columns:
            df['round_epochs'] = df['epochs']
        elif 'local_epochs' in df.columns:
            df['round_epochs'] = df['local_epochs']
    return df

def average_runs(dfs):
    if len(dfs) == 1:
        return dfs[0]
    max_r = max(len(df) for df in dfs)
    records = []
    for r_idx in range(max_r):
        rnd = r_idx + 1
        active_dfs = [df for df in dfs if len(df) >= rnd]
        act_mat = 'A' if rnd % 2 == 1 else 'B'
        
        accs = []
        for df in dfs:
            if len(df) >= rnd:
                accs.append(df['accuracy'].iloc[r_idx])
            else:
                accs.append(df['accuracy'].iloc[-1])
        acc = float(np.mean(accs))
        
        ep = float(np.mean([df['round_epochs'].iloc[r_idx] for df in active_dfs]))
        var = float(np.mean([df['variance'].iloc[r_idx] for df in active_dfs]))
        
        row = {
            'round': rnd,
            'active_matrix': act_mat,
            'accuracy': acc,
            'round_epochs': ep,
            'variance': var,
        }
        records.append(row)
    return pd.DataFrame(records)

def generate_section48_table():
    s48_configs = [
        ('MNLI ($C=20, R=50, \\alpha=1000$)', 'mnli', 20, 50, [0.65, 0.72]),
        ('MNLI ($C=50, R=30, \\alpha=1000$)', 'mnli', 50, 30, [0.55, 0.65]),
        ('QNLI ($C=20, R=50, \\alpha=1000$)', 'qnli', 20, 50, [0.80, 0.83]),
        ('QNLI ($C=50, R=30, \\alpha=1000$)', 'qnli', 50, 30, [0.75, 0.82]),
        ('QQP ($C=20, R=50, \\alpha=1000$)',  'qqp',  20, 50, [0.65, 0.74]),
        ('QQP ($C=50, R=30, \\alpha=1000$)',  'qqp',  50, 30, [0.75, 0.78]),
        ('SST-2 ($C=20, R=50, \\alpha=1000$)', 'sst2', 20, 50, [0.85, 0.90]),
        ('SST-2 ($C=50, R=30, \\alpha=1000$)', 'sst2', 50, 30, [0.85, 0.90]),
    ]

    lines = [
        '| Benchmark Setting | Method Configuration | Seeds | Peak Acc | Milestone 1 (Time, Cost) | Milestone 2 (Time, Cost) | Total Rounds | Complete MB | Steps | Mean Ep/Rd | Ep (A) | Ep (B) |',
        '| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |'
    ]

    for title, ds, c, r, mstones in s48_configs:
        rate = 4.6875 if c == 20 else 11.71875
        methods = [
            ('FDA Pure E1TO20 [Emax/1]', '#15803d', f'results/round_metrics/*THESIS-RBASE-PURE-DS-{ds}-C{c}-*-M-fda-pure-LR0.1-RK4-*CRATIO1-*.csv'),
            ('FDA Pure E1TO20 [Emax/2]', '#059669', f'results/round_metrics/*THESIS-RBASE-PURE-DS-{ds}-C{c}-*-M-fda-pure-LR0.1-RK4-*CRATIO0.5-*.csv'),
            ('FDA E3TO20 [Emax/1-NODISC]', '#0369a1', f'results/round_metrics/*CALIB-DS-{ds}-C{c}-R{r}-M-fda-calib-LR0.1*E3TO20*CRATIO1*NODISC*.csv'),
            ('FDA E3TO20 [Emax/1-BDISC]', '#1e40af', f'results/round_metrics/*CALIB-DS-{ds}-C{c}-R{r}-M-fda-calib-LR0.1*E3TO20*CRATIO1*BDISC3*.csv'),
            ('FDA E3TO20 [Emax/2-Old]', '#4338ca', f'results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-fda-standard-LR0.1-RK4-E3TO20-*.csv'),
            ('RoLoRA E20', '#b91c1c', f'results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-rolora-LR0.1-RK4-S*-E20*.csv'),
            ('RoLoRA E10', '#b91c1c', f'results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-rolora-LR0.1-RK4-S*-E10*.csv'),
            ('RoLoRA E5', '#b91c1c', f'results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-rolora-LR0.1-RK4-S*-E5*.csv'),
        ]
        rows = []
        for mname, col, pat in methods:
            files = sorted(glob.glob(pat))
            if not files:
                continue
            dfs = [prep(pd.read_csv(f)) for f in files]
            df_avg = average_runs(dfs)
            max_acc = float(df_avg['accuracy'].max())
            tot_r = len(df_avg)
            tot_steps = int(round(np.mean([df['round_steps'].sum() for df in dfs]))) if 'round_steps' in dfs[0].columns else 0
            mean_ep = float(np.mean([df['round_epochs'].mean() for df in dfs]))
            
            ea_list, eb_list = [], []
            for df in dfs:
                if 'active_matrix' in df.columns:
                    ea_list.append(df[df['active_matrix'] == 'A']['round_epochs'].mean() if len(df[df['active_matrix'] == 'A']) > 0 else 0.0)
                    eb_list.append(df[df['active_matrix'] == 'B']['round_epochs'].mean() if len(df[df['active_matrix'] == 'B']) > 0 else 0.0)
                else:
                    ea_list.append(df['round_epochs'].mean())
                    eb_list.append(df['round_epochs'].mean())
            mean_ea = float(np.mean(ea_list))
            mean_eb = float(np.mean(eb_list))
            
            m_cross = []
            for th in mstones:
                cross = np.where(df_avg['accuracy'].values >= th)[0]
                if len(cross) > 0:
                    rhit = int(cross[0]) + 1
                    th_time = float(np.mean([df.loc[:rhit-1, 'time_round_total'].sum() if len(df) >= rhit else df['time_round_total'].sum() for df in dfs])) / 3600.0
                    th_mb = rhit * rate
                    m_cross.append(f'R{rhit} ({th_time:.2f} h, {th_mb:.1f} MB)')
                else:
                    m_cross.append('—')
            rows.append({
                'mname': mname,
                'col': col,
                'num_seeds': len(dfs),
                'max_acc': max_acc,
                'm1': m_cross[0],
                'm2': m_cross[1],
                'tot_r': tot_r,
                'mb': tot_r * rate,
                'steps': tot_steps,
                'ep': mean_ep,
                'ea': mean_ea,
                'eb': mean_eb
            })
        best_acc = max(r_item['max_acc'] for r_item in rows)
        for idx, r_item in enumerate(rows):
            bset_cell = f'**{title}**' if idx == 0 else ''
            c_tag = r_item['col']
            m_tag = r_item['mname']
            if abs(r_item['max_acc'] - best_acc) < 1e-4:
                m_cell = f'<span style="color: {c_tag};">**{m_tag}**</span>'
                acc_cell = f'**{r_item["max_acc"]:.4f}**'
            else:
                m_cell = f'<span style="color: {c_tag};">{m_tag}</span>'
                acc_cell = f'{r_item["max_acc"]:.4f}'
            line = f'| {bset_cell} | {m_cell} | {r_item["num_seeds"]} | {acc_cell} | {r_item["m1"]} | {r_item["m2"]} | {r_item["tot_r"]} | {r_item["mb"]:.1f} MB | {r_item["steps"]:,} | {r_item["ep"]:.2f} | {r_item["ea"]:.2f} | {r_item["eb"]:.2f} |'
            lines.append(line)

    return '\n'.join(lines)

def generate_section49_table():
    noniid_groups = [
        ('MNLI ($C=20, \\alpha=0.5$)', 'mnli', 20, [0.65, 0.72, 0.74]),
        ('QNLI ($C=20, \\alpha=0.5$)', 'qnli', 20, [0.80, 0.83, 0.84]),
        ('MNLI ($C=50, \\alpha=0.5$)', 'mnli', 50, [0.50, 0.53, 0.55]),
        ('QNLI ($C=50, \\alpha=0.5$)', 'qnli', 50, [0.55, 0.60, 0.65]),
    ]

    lines = [
        '| Benchmark Setting | Method Configuration | Peak Acc (Best Rd) | Final Acc | Targeted Milestone 1 (Time, Complete MB) | Targeted Milestone 2 (Time, Complete MB) | Targeted Milestone 3 (Time, Complete MB) | Total Rounds | Complete MB | Steps | Wall-Clock Time (s / h) | Mean Ep/Rd | <span style="color: #047857;">Mean Ep (A)</span> | <span style="color: #7c3aed;">Mean Ep (B)</span> | Mean Drift Var $\\bar{\\nu}$ | Max Drift Var $\\nu_{\\max}$ |',
        '| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---:'
    ]

    for title, ds, c, mstones in noniid_groups:
        rate = 4.6875 if c == 20 else 11.71875
        if c == 20:
            methods = [
                ('FDA Pure E1TO40 [Emax/2, k=0.5]', '#15803d', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO0.5-*.csv'),
                ('RoLoRA E10', '#b91c1c', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C20-*-M-rolora-E10-LR0.1-RK4-*.csv'),
                ('FDA Pure E1TO20 [Emax/2, k=0.5]', '#059669', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv'),
                ('RoLoRA E20', '#b45309', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C20-*-M-rolora-E20-LR0.1-RK4-*.csv'),
                ('FDA Pure E1TO20 [Emax/1, k=1.0]', '#15803d' if ds == 'mnli' else '#0284c7', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv'),
            ]
            if ds == 'qnli':
                methods.append(('FDA Pure E1TO40 [Emax/1, k=1.0]', '#0047AB', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO1-*.csv'))
        else:
            methods = [
                ('FDA Pure E1TO20 [Emax/1, k=1.0]', '#15803d' if ds == 'mnli' else '#0284c7', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv'),
                ('FDA Pure E1TO20 [Emax/2, k=0.5]', '#059669', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv'),
                ('RoLoRA E10', '#b91c1c', f'results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C50-*-M-rolora-E10-LR0.1-RK4-*.csv'),
            ]

        rows = []
        for mname, col, pat in methods:
            files = sorted(glob.glob(pat))
            if not files:
                continue
            dfs = [prep(pd.read_csv(f)) for f in files]
            df_avg = average_runs(dfs)
            max_acc = float(df_avg['accuracy'].max())
            best_rnd = int(df_avg.loc[df_avg['accuracy'].idxmax(), 'round'])
            final_acc = float(df_avg['accuracy'].iloc[-1])
            tot_r = len(df_avg)
            tot_time = float(np.mean([df['time_round_total'].sum() for df in dfs])) if 'time_round_total' in dfs[0].columns else 0.0
            tot_steps = int(round(np.mean([df['round_steps'].sum() for df in dfs]))) if 'round_steps' in dfs[0].columns else 0
            mean_ep = float(np.mean([df['round_epochs'].mean() for df in dfs]))
            
            ea_list, eb_list = [], []
            for df in dfs:
                if 'active_matrix' in df.columns:
                    ea_list.append(df[df['active_matrix'] == 'A']['round_epochs'].mean() if len(df[df['active_matrix'] == 'A']) > 0 else 0.0)
                    eb_list.append(df[df['active_matrix'] == 'B']['round_epochs'].mean() if len(df[df['active_matrix'] == 'B']) > 0 else 0.0)
                else:
                    ea_list.append(df['round_epochs'].mean())
                    eb_list.append(df['round_epochs'].mean())
            mean_ea = float(np.mean(ea_list))
            mean_eb = float(np.mean(eb_list))
            
            m_var = float(np.mean([df['variance'].mean() for df in dfs])) if 'variance' in dfs[0].columns else 0.0
            max_var = float(np.mean([df['variance'].max() for df in dfs])) if 'variance' in dfs[0].columns else 0.0
            
            m_cross = []
            for th in mstones:
                cross = np.where(df_avg['accuracy'].values >= th)[0]
                if len(cross) > 0:
                    rhit = int(cross[0]) + 1
                    th_time = float(np.mean([df.loc[:rhit-1, 'time_round_total'].sum() if len(df) >= rhit else df['time_round_total'].sum() for df in dfs])) / 3600.0
                    th_mb = rhit * rate
                    m_cross.append(f'R{rhit} ({th_time:.2f} h, {th_mb:.1f} MB)')
                else:
                    m_cross.append('—')

            rows.append({
                'mname': mname,
                'col': col,
                'num_seeds': len(dfs),
                'peak_acc': max_acc,
                'best_rnd': best_rnd,
                'final_acc': final_acc,
                'm1': m_cross[0],
                'm2': m_cross[1],
                'm3': m_cross[2],
                'tot_r': tot_r,
                'mb': tot_r * rate,
                'steps': tot_steps,
                'time_s': tot_time,
                'time_h': tot_time / 3600.0,
                'ep': mean_ep,
                'ea': mean_ea,
                'eb': mean_eb,
                'var_mean': m_var,
                'var_max': max_var,
            })

        best_acc = max(r_item['peak_acc'] for r_item in rows)
        for idx, r_item in enumerate(rows):
            bset_cell = f'**{title}**' if idx == 0 else ''
            c_tag = r_item['col']
            m_tag = r_item['mname']
            
            if abs(r_item['peak_acc'] - best_acc) < 1e-4:
                m_cell = f'<span style="color: {c_tag};">**{m_tag}**</span>'
                peak_cell = f'**{r_item["peak_acc"]:.4f}** (R{r_item["best_rnd"]})'
            else:
                m_cell = f'<span style="color: {c_tag};">{m_tag}</span>'
                peak_cell = f'{r_item["peak_acc"]:.4f} (R{r_item["best_rnd"]})'
                
            final_cell = f'{r_item["final_acc"]:.4f}'
            time_cell = f'{r_item["time_s"]:,.0f} s ({r_item["time_h"]:.2f} h)'
            steps_cell = f'{r_item["steps"]:,}'
            ep_cell = f'**{r_item["ep"]:.2f}**' if 'FDA Pure' in m_tag else f'{r_item["ep"]:.2f}'
            ea_cell = f'<span style="color: #047857; background-color: rgba(4, 120, 87, 0.08); padding: 1px 5px; border-radius: 3px;">{r_item["ea"]:.2f}</span>'
            eb_cell = f'<span style="color: #7c3aed; background-color: rgba(124, 58, 237, 0.08); padding: 1px 5px; border-radius: 3px;">{r_item["eb"]:.2f}</span>'
            
            line = f'| {bset_cell} | {m_cell} | {peak_cell} | {final_cell} | {r_item["m1"]} | {r_item["m2"]} | {r_item["m3"]} | {r_item["tot_r"]} | {r_item["mb"]:.1f} MB | {steps_cell} | {time_cell} | {ep_cell} | {ea_cell} | {eb_cell} | {r_item["var_mean"]:.3f} | {r_item["var_max"]:.3f} |'
            lines.append(line)

    return '\n'.join(lines)

if __name__ == '__main__':
    t48 = generate_section48_table()
    t49 = generate_section49_table()
    print("Generated Section 4.8 Table:")
    print(t48[:500])
    print("\nGenerated Section 4.9 Table:")
    print(t49[:500])
