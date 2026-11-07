"""Aggregate-only report and scientific figure. Never render patient frames."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ORDER = ('LEGACY_E2E', 'LEGACY_WINDOW', 'SB0', 'SB1', 'SB2', 'SB3', 'doctor02', 'doctor03', 'MOCK100')
NAMES = {'LEGACY_E2E': '旧版完整流程', 'LEGACY_WINDOW': '参考窗口加旧版规则', 'MOCK100': '接口固定100分'}


def f(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def pct(value):
    return '—' if value is None else f'{value * 100:.1f}%'


def comparison(results, first, second):
    key = f'{first}__minus__{second}'
    if key in results['paired_comparisons']:
        return results['paired_comparisons'][key]
    row = results['paired_comparisons'][f'{second}__minus__{first}'].copy()
    row['first_mae'], row['second_mae'] = row.get('second_mae'), row.get('first_mae')
    if row['mae_difference'] is not None:
        row['mae_difference'] *= -1
        row['ci95_video_bootstrap_descriptive'] = [-v for v in row['ci95_video_bootstrap_descriptive'][::-1]]
    return row


def plot(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = results['methods']
    labels = ['Legacy E2E', 'Window + legacy', 'SB0 mode', 'SB1 geometry', 'SB2 appearance', 'SB3 fusion', 'Doctor 02', 'Doctor 03', 'Mock 100']
    colors = ['#a45d4e', '#c39772', '#a3adb6', '#237b78', '#769eb2', '#3c698b', '#72649b', '#947caa', '#b8b8b8']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(16, 11), layout='constrained')
        fig.suptitle(f"Sunnybrook comparison | matched numeric cohort n={results['common_numeric_count']} | reference: Doctor 01", fontsize=16)
        ax = axes[0, 0]
        for i, name in enumerate(ORDER):
            metric = methods[name]['common_numeric_metrics']
            if metric:
                low, high = metric['mae_ci95_video_bootstrap_descriptive']
                ax.barh(i, metric['total_mae'], color=colors[i])
                ax.errorbar(metric['total_mae'], i, xerr=[[metric['total_mae'] - low], [high - metric['total_mae']]], fmt='none', color='#263744', capsize=3)
        ax.set(yticks=range(len(ORDER)), yticklabels=labels, xlabel='Total MAE (points), descriptive video bootstrap interval', title='A  Same videos for every row')
        ax.invert_yaxis()
        ax = axes[0, 1]
        positions = np.arange(len(ORDER))
        ax.barh(positions - .18, [methods[n]['numeric_coverage'] * 100 for n in ORDER], height=.35, color='#73a5a5', label='13 numeric outputs')
        ax.barh(positions + .18, [methods[n]['valid_coverage'] * 100 for n in ORDER], height=.35, color='#294f6d', label='All legal + total 0..100')
        ax.set(yticks=positions, yticklabels=labels, xlabel='Percent of accessible input videos', xlim=(0, 105), title='B  Coverage before restricting the cohort')
        ax.invert_yaxis(); ax.legend(loc='upper left', bbox_to_anchor=(0, -.13), ncol=2, fontsize=9)
        ax = axes[1, 0]
        metrics = [methods[n]['common_numeric_metrics'] for n in ORDER]
        if all(metrics):
            items = list(metrics[0]['subscores'])
            heat = np.array([[m['subscores'][k]['normalized_mae'] for k in items] for m in metrics])
            display = ax.imshow(heat, aspect='auto', vmin=0, vmax=max(.5, float(heat.max())), cmap='YlOrRd')
            fig.colorbar(display, ax=ax, shrink=.8, label='MAE / legal item range')
            ax.set(yticks=range(len(ORDER)), yticklabels=labels, xticks=range(13), xticklabels=['R-eye','R-cheek','R-mouth','V-brow','V-eye','V-smile','V-snarl','V-pucker','S-brow','S-eye','S-smile','S-snarl','S-pucker'], title='C  Item errors on the matched cohort')
            ax.tick_params(axis='x', labelrotation=65, labelsize=8)
        ax = axes[1, 1]
        if all(metrics):
            recall = [m['synkinesis_macro_recall'] * 100 if m['synkinesis_macro_recall'] is not None else np.nan for m in metrics]
            false_alarms = [m['synkinesis_macro_fpr'] * 100 if m['synkinesis_macro_fpr'] is not None else np.nan for m in metrics]
            ax.barh(positions - .18, recall, height=.35, color='#398b82', label='Nonzero recall')
            ax.barh(positions + .18, false_alarms, height=.35, color='#c69a78', label='False positive rate')
        ax.set(yticks=positions, yticklabels=labels, xlabel='Macro average across five actions (%)', xlim=(0, 110), title='D  Synkinesis: recall and false alarms together')
        ax.invert_yaxis(); ax.legend(loc='lower right', fontsize=9)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, facecolor='white')
        fig.savefig(path.with_suffix('.pdf'), facecolor='white')
        plt.close(fig)


def write_report(results, path, output):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p, methods = results['protocol'], results['methods']
    common = results['common_numeric_count']
    conclusion = ''
    if common:
        old = methods['LEGACY_E2E']['common_numeric_metrics']
        sb1 = methods['SB1']['common_numeric_metrics']
        conclusion = f"在全部方法共同完成的 {common} 段视频上，旧版完整流程总分 MAE 为 {f(old['total_mae'])} 分，平均偏差为 {f(old['bias'])} 分；SB1 的 MAE 为 {f(sb1['total_mae'])} 分。完整结果还需结合下面的输出覆盖率、分项误差与联动误报率判读。"
    lines = ['# Sunnybrook 统一评价矩阵与方法比较', '',
        conclusion, '',
        f"评价日期：{p['date']}。doctor01 的 13 个分项为 ground truth，总分统一由分项公式计算。纳入历史 {p['target_count']} 段评估视频，当前可访问 {p['accessible_count']} 段。每段视频作为一次独立评估；重复患者的不同访视可能跨折，因此属于访视层面开发验证。", '',
        '旧版完整流程、旧规则的取帧诊断、SB0 至 SB3、两位年轻医生及接口固定满分对照采用同一指标实现。所有旧版源文件保持原样，不修正除零、取帧或评分规则。', '',
        '## 统一评价矩阵', '',
        '| 评价维度 | 主要指标 | 同时检查 | 判读规则 |',
        '| --- | --- | --- | --- |',
        '| 输入与输出覆盖 | 数值输出率和合法输出率 | 缺失视频、失败类别、分项越界、总分越界 | 与误差同时报告，失败不补成正常分 |',
        '| 总分误差 | MAE 及描述性重采样区间 | RMSE、平均偏差、绝对一致性 ICC A,1、Bland Altman 界限 | 统一在相同视频上比较，不用相关系数代替一致性 |',
        '| 三大维度 | 静态、自主运动、联动各自归一化 MAE | 维度加权分的 MAE | 三维度等权，避免五个自主运动项掩盖其他维度 |',
        '| 十三个分项 | 分项 MAE 与二次加权 Kappa | 完全一致率、相差不超过一级、现有真值等级的平衡召回率与宏 F1、混淆矩阵 | 同时列等级样本数；未出现等级不假定已学会 |',
        '| 非零联动 | 各动作召回率和误报率 | 精确率、F1、阳性例数，五动作宏平均 | 不用大量零分形成的高准确率说明检出能力 |',
        '| 配对比较 | 相同视频的绝对误差差值 | SB1 对其他方法、算法对两位医生 | 每次配对使用同一行索引，区间跨零不宣称优越 |',
        '| 误差分层 | 按专家总分区间的 MAE | 每层样本数；后续可接入人工质量标签 | 稀疏层仅描述，不设未验证的临床阈值 |',
        '| 运行成本 | 旧版工作进程单视频中位耗时与 P95 | 模型初始化与取帧在内，两个并发任务 | 研究特征缓存与新跑旧版的耗时不直接排名 |', '',
        '合法分项范围依次为静态 0–1、0–2、0–1，自主运动各 1–5，联动各 0–3。总分为 4 × 自主运动和 − 5 × 静态和 − 联动和。完整数值预测即使越界也保留并计入原始误差；另行统计合法输出率。分项混淆矩阵保留非法等级，避免丢掉旧版输出的自主运动零分。', '',
        '## 固定实验协议', '',
        '- 使用原固定外层五折、内层三折和 seed 20261012，SB0 至 SB3 从现有特征重新训练，逐项核对原有折外预测。标准化、PCA 与参数选择只使用各自训练折。',
        '- 学习算法保留历史 68 段训练流程；当前共同输入队列在运行旧版之前按源文件可访问性冻结。缺失视频的历史特征只用于复现原实验，不伪造其旧版预测。',
        '- 三位医生 CSV 指纹及可访问视频 SHA-256 均与历史记录核对；doctor02、doctor03 只作对照。缓存、折号、代码指纹和逐视频结果保留在本地 artifacts。',
        '- 旧版完整流程保留整段视频几何极值选帧、原始颜色输入和静息帧索引。参考窗口诊断用既有提取器选出的静息中点及动作峰值时间，经 OpenCV 定位取帧，再执行同一个旧版 IMAGE 评分器；它不代表人工动作真值。旧代码使用当前研究依赖环境运行，不声称复原当年未锁定的部署环境；environment.json 记录实际版本，并阻止跨环境混用同一输出目录的缓存。',
        '- 不使用全量训练后的导出模型评价训练视频，不调整阈值，不裁剪分项或总分，不根据测试结果重新分折。固定100分只作为占位对照。',
        '- 配对视频 bootstrap 5000 次。区间为描述性结果，不包含重复患者依赖及重新训练的不确定性，不作为正式显著性检验。5、10、15 分误差带仅为工程描述，不是临床等效阈值。', '',
        '## 可访问视频的完成率', '',
        '| 方法 | 输入数 | 数值输出 | 全部合法 | 非法分项行 | 总分越界行 | 原始 MAE及其样本数 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for name in ORDER:
        entry = methods[name]; m = entry['own_numeric_metrics']
        lines.append(f"| {NAMES.get(name, name)} | {entry['input_count']} | {entry['numeric_count']} | {entry['valid_count']} | {m['invalid_item_rows'] if m else '—'} | {m['invalid_raw_total_rows'] if m else '—'} | {f(m['total_mae']) if m else '—'}，n={entry['numeric_count']} |")
    lines += ['', '本表最后一列可能采用不同子集，用于了解各方法实际产出，不据此进行跨方法排名。全体方法共同有数值输出的视频结果如下。', '',
        f'## 全部方法共同视频比较', '', f'共同队列 n={common}。下表所有行使用完全相同的视频；原始越界值保留。', '',
        '| 方法 | 总分 MAE | 描述性95%区间 | RMSE | 偏差 | ICC A,1 | 三维等权归一化MAE | 联动召回率 | 联动误报率 |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ORDER:
        m = methods[name]['common_numeric_metrics']
        if m:
            low, high = m['mae_ci95_video_bootstrap_descriptive']
            lines.append(f"| {NAMES.get(name, name)} | {f(m['total_mae'])} | [{f(low)}, {f(high)}] | {f(m['total_rmse'])} | {f(m['bias'])} | {f(m['icc_absolute_single'],3)} | {f(m['equal_dimension_normalized_mae'],3)} | {pct(m['synkinesis_macro_recall'])} | {pct(m['synkinesis_macro_fpr'])} |")
    if not common:
        lines += ['', '全体交集为空，不能生成这一比较；应使用下面预先定义的逐方法配对比较，并同时报告覆盖率。']
    lines += ['', '联动召回率和误报率为五个动作宏平均；逐动作的阳性数、混淆矩阵、总分误差带、分层结果和全部13个分项位于本地 results.json。', '',
        '## 三大维度误差', '',
        '同一共同队列上的归一化分项 MAE。每列以各分项合法跨度归一化后在该维度取平均，越小越好。', '',
        '| 方法 | 静态 | 自主运动 | 联动 |', '| --- | ---: | ---: | ---: |']
    for name in ORDER:
        m = methods[name]['common_numeric_metrics']
        if m:
            d = m['dimension_normalized_mae']
            lines.append(f"| {NAMES.get(name,name)} | {f(d['rest'],3)} | {f(d['voluntary'],3)} | {f(d['synkinesis'],3)} |")
    lines += ['', '## 十三个分项误差', '', '同一共同队列上的原始等级 MAE。各分项等级跨度不同，跨分项比较时应结合上表及汇总图。', '',
        '| 分项 | 旧完整 | 窗口旧规则 | SB0 | SB1 | SB2 | SB3 | 医生02 | 医生03 | 固定100 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    if common:
        item_labels = ['静态眼','静态鼻唇沟','静态嘴','抬眉运动','闭眼运动','微笑运动','呲牙运动','嘟嘴运动','抬眉联动','闭眼联动','微笑联动','呲牙联动','嘟嘴联动']
        items = list(methods['SB1']['common_numeric_metrics']['subscores'])
        for item, label in zip(items, item_labels):
            row = [f(methods[n]['common_numeric_metrics']['subscores'][item]['mae'],3) for n in ORDER]
            lines.append('| '+label+' | '+' | '.join(row)+' |')
    lines += ['',
        '## SB1 的配对比较', '',
        '| 对照方法 | 共同视频数 | SB1 MAE | 对照MAE | SB1减对照误差 | 描述性95%区间 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for name in ORDER:
        if name == 'SB1': continue
        row = comparison(results, 'SB1', name)
        ci = row['ci95_video_bootstrap_descriptive']
        lines.append(f"| {NAMES.get(name,name)} | {row['count']} | {f(row.get('first_mae'))} | {f(row.get('second_mae'))} | {f(row['mae_difference'])} | {'['+f(ci[0])+', '+f(ci[1])+']' if ci else '—'} |")
    lines += ['', '差值小于零表示 SB1 误差较小。不同配对行的队列可能不同，不能用这些差值做传递排名。', '',
        '## 旧版失败与设计限制', '']
    for name in ORDER[:2]:
        entry=methods[name]
        lines.append(f"- {NAMES[name]}：数值完成 {entry['numeric_count']}/{entry['input_count']}，合法完成 {entry['valid_count']}/{entry['input_count']}；失败类别 {entry['failure_type_counts']}。")
    lines += ['- 静态鼻唇沟在旧版代码中始终为0；自主运动用左右比例映射，不能区分两侧运动均很小和两侧运动充分；联动由非目标几何比例经验公式产生。',
        '- 全局极值取帧没有动作分段约束；原版未检查每帧是否检测到人脸，多个比例未保护零分母。参考窗口诊断有助于区分选帧问题和评分规则问题，但没有隔离 OpenCV 定位、JPEG 与图像检测误差。',
        '- SB1 至 SB3 的动作窗口仍是六等分假设；尚未接入标注工具的人工动作边界。SB1 在这批数据的较低误差不代表已经验证对新患者或新医院的效果。', '',
        '## 本机旧版运行耗时', '',
        '| 方法 | 已运行视频数 | 中位秒数 | P95秒数 |', '| --- | ---: | ---: | ---: |']
    for name in ORDER[:2]:
        t = methods[name]['runtime_seconds']
        if t: lines.append(f"| {NAMES[name]} | {t['count']} | {f(t['median'])} | {f(t['p95'])} |")
    lines += ['', '耗时包括旧版模型初始化、选帧和评分，按两个并发任务记录；运行失败任务也计入。参考窗口诊断使用预先计算的峰值时间，表内不包含这些时间的提取成本，因此不是可直接部署的完整视频延迟。SB0 至 SB3 本次复用历史特征，未分别测量端到端耗时，不据此进行性能排名。', '',
        '## 结果解释与后续顺序', '',
        '旧版完整流程在共同视频上的总分误差高于训练折众数基线，并存在明显高估。参考窗口取帧能够降低一部分误差，但旧规则的分项和联动表现仍差。高联动召回率同时伴随大量误报，不能解释为较好的联动识别。', '',
        'SB1 是这批开发数据中总分 MAE 最低的学习算法，适合作为后续优化基线；SB1 与 SB3 的配对区间跨零，尚不能确认两者差异稳定。SB1 与两位年轻医生的总分误差接近，但配对区间也均跨零，且三维度等权归一化分项误差仍高于两位医生，不能据总分相近宣称已达到或超过医生整体水平。', '',
        '后续先将人工动作区间接入特征提取，检查静态鼻唇沟与非目标区域联动特征，再在同一评价协议下增加新版本。最终模型冻结后，用新患者视频做独立测试；现有视频访视队列用于开发比较。', '',
        '## 历史完整队列复现', '',
        'SB0 至 SB3 的重新训练预测与原始折外文件逐项一致。下表保留历史68段结果，不能将它与不同队列的旧版单行直接排名。', '',
        '| 方法 | 视频数 | 总分 MAE | RMSE | 偏差 | ICC A,1 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, m in results['historical_68_metrics'].items():
        lines.append(f"| {NAMES.get(name,name)} | {m['count']} | {f(m['total_mae'])} | {f(m['total_rmse'])} | {f(m['bias'])} | {f(m['icc_absolute_single'],3)} |")
    lines += ['', '## 汇总图', '', '![全部方法共同视频的误差、覆盖率、分项和联动诊断](unified-evaluation.png)', '',
        '## 复现入口与结果文件', '',
        '使用仓库 algorithm 下各版本及 shared/fphelper 的研究代码，src/fphelper 保留兼容导入层，tools/benchmark 提供统一评估入口。评估依赖见 pyproject.toml；旧版另需 OpenCV 和 MediaPipe。命令示例：', '',
        '```sh', 'PYTHONPATH=src python tools/benchmark/run.py \\', '  --source /absolute/path/to/artifacts/first-validation \\', '  --data-root /absolute/path/to/fp \\', '  --output artifacts/unified-evaluation-20261107', '```', '',
        '输出的 protocol.json 固定输入与评价规则，results.json 保存全部矩阵指标，predictions.csv 和 predictions.npz 保存匿名逐视频结果，status.json 保存失败与耗时，training-folds.json 保存嵌套验证记录，legacy-cache 按视频和代码指纹缓存结果。临时 JPEG 由父进程拥有的临时目录清理，包括工作进程失败或超时。', '',
        '本次使用已有研究环境运行，未把虚拟环境或原始患者视频复制到桌面仓库。完整结果位于仓库 artifacts/unified-evaluation-20261107。', '',
        '## 方法依据', '',
        '参数选择与外层性能评估分开的原则见 [scikit-learn 嵌套验证](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。分项二次加权 Kappa 的实现定义见 [scikit-learn 指标文档](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)。', '']
    path.write_text('\n'.join(lines))
    plot(results, path.with_suffix('.png'))
    with (output / 'matrix.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method','input_count','numeric_count','valid_count','common_numeric_count','mae_common','rmse_common','bias_common','icc_absolute_common','normalized_mae_common','syn_recall_common','syn_fpr_common'])
        for name in ORDER:
            entry=methods[name]; m=entry['common_numeric_metrics'] or {}
            writer.writerow([name,entry['input_count'],entry['numeric_count'],entry['valid_count'],common,*[m.get(k) for k in ('total_mae','total_rmse','bias','icc_absolute_single','equal_dimension_normalized_mae','synkinesis_macro_recall','synkinesis_macro_fpr')]])
