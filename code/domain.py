import utils
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List
from scipy.ndimage import label
from collections import defaultdict
from scipy.stats import mannwhitneyu
from torch_scatter import scatter_mean, scatter_sum

class NodeDomainAnnotator:
    def __init__(self, evalue_threshold=1e-1):
        annotation_df = "/mnt/disk1/nwn/data/uniprotkb/interproscan_pfam_output.tsv"
        columns = ['Sequence ID', 'MD5 Checksum', 'Sequence Length', 'Analysis Method',
                    'Accession Number', 'Domain Name', 'Start Position', 'End Position',
                    'E-value', 'Status', 'Date', 'InterPro Entry ID', 'InterPro Entry Description',
                    'GO Terms', 'Pathway Entry']
        scan_df = pd.read_csv(annotation_df, sep='\t', names=columns, dtype=str)
        scan_df['Uniprot ID'] = scan_df['Sequence ID'].apply(lambda x: x.split('|')[1] if '|' in x else x)
        
        if evalue_threshold is not None:
            scan_df['E-value'] = pd.to_numeric(scan_df['E-value'], errors='coerce')
            scan_df = scan_df[scan_df['E-value'] < evalue_threshold]
        
        self.scan_df = scan_df

    def save_ids(self):
        # 提取 Uniprot ID 列，去重
        unique_ids = self.scan_df['Uniprot ID'].dropna().unique()

        output_path = "/mnt/disk1/nwn/data/uniprotkb/pfam_ids.txt"
        # 保存为 txt 文件，每行一个 ID
        with open(output_path, 'w') as f:
            for uid in unique_ids:
                f.write(uid + '\n')

    def get_coarse_domain_matrix(self, prot_lens: Dict[str, int]) -> torch.Tensor:
        """
        返回所有蛋白粗粒度的结构域有无标注，[sum(seq_len_list), 1]
        """
        masks = []
        for prot_idx, seq_len in prot_lens.items():
            mask = torch.zeros(seq_len, 1)
            annots = self.scan_df[self.scan_df['Uniprot ID'] == prot_idx]
            for _, row in annots.iterrows():
                try:
                    start = int(row['Start Position']) - 1
                    end = int(row['End Position'])
                    mask[start:end] = 1
                except:
                    continue
            masks.append(mask)
        return torch.cat(masks, dim=0)

    def get_specific_domain_matrix(self, prot_lens: Dict[str, int], target_domains: List[str]) -> torch.Tensor:
        """
        返回指定结构域的节点注释，[sum(seq_len_list), len(target_domains)]
        """
        masks = []
        target_domains_dict = {domain: idx for idx, domain in enumerate(target_domains)}
        for prot_idx, seq_len in prot_lens.items():
            mask = torch.zeros(seq_len, len(target_domains))
            annots = self.scan_df[self.scan_df['Uniprot ID'] == prot_idx]
            for _, row in annots.iterrows():
                domain_name = row['Accession Number']
                if domain_name not in target_domains:
                    continue
                try:
                    start = int(row['Start Position']) - 1
                    end = int(row['End Position'])
                    domain_idx = target_domains_dict[domain_name]
                    mask[start:end, domain_idx] = 1
                except:
                    continue
            masks.append(mask)
        return torch.cat(masks, dim=0)
    
def cfs_region(activation_path, annotator, target_domains=None, device="cuda"):
    data = torch.load(activation_path, weights_only=True)
    sparse_acts = data["sparse_acts"].coalesce()
    # prot_idx_list = data["prot_idx"]
    # seq_len_list = data["seq_len"]
    prot_lens = data["prot_lens"]


    nonzero_idx = sparse_acts.indices().T  # [nnz，2]
    nonzero_val = sparse_acts.values()
    domain_matrix = annotator.get_coarse_domain_matrix(prot_lens)
    coarse_mask = domain_matrix.sum(dim=1).bool() 
    if target_domains is not None:
        specific_matrix = annotator.get_specific_domain_matrix(prot_lens, target_domains)
        specific_mask = specific_matrix.sum(dim=1).bool()
        domain_mask = specific_mask
        background_mask = coarse_mask & ~specific_mask
        del specific_matrix
    else:
        domain_mask = coarse_mask
        background_mask = ~coarse_mask

    del data, domain_matrix

    # 统计每条蛋白的起止位置
    seq_boundaries = []
    cur = 0
    for length in prot_lens.values():
        seq_boundaries.append((cur, cur + length))
        cur += length

    # 收集每条蛋白的结构域片段和非结构域片段
    region_info = []
    for (start, end) in seq_boundaries:
        domain_seg = domain_mask[start:end].cpu().numpy()
        neg_seg = background_mask[start:end].cpu().numpy()

        # 标注连续结构域区域
        pos_labeled, n_pos = label(domain_seg)
        neg_labeled, n_neg = label(neg_seg)

        pos_regions = [torch.arange(start + (pos_labeled == i).nonzero()[0][0],
                                    start + (pos_labeled == i).nonzero()[-1][0] + 1) 
                       for i in range(1, n_pos+1)]

        neg_regions = [torch.arange(start + (neg_labeled == i).nonzero()[0][0],
                                    start + (neg_labeled == i).nonzero()[-1][0] + 1)
                       for i in range(1, n_neg+1)]

        region_info.append((pos_regions, neg_regions))

    del domain_seg, neg_seg, pos_labeled, neg_labeled, pos_regions, neg_regions

    # 在region_info构建后
    pos_region_map = []  # 每个 pos region 的 node index
    neg_region_map = []  # 每个 neg region 的 node index

    for pos_regions, neg_regions in region_info:
        pos_region_map.extend(pos_regions)
        neg_region_map.extend(neg_regions)

    pos_all_nodes = torch.cat(pos_region_map).to(device)
    pos_region_ids = torch.cat([torch.full_like(r, i) for i, r in enumerate(pos_region_map)]).to(device)
    neg_all_nodes = torch.cat(neg_region_map).to(device)
    neg_region_ids = torch.cat([torch.full_like(r, i) for i, r in enumerate(neg_region_map)]).to(device)

    del region_info, pos_region_map, neg_region_map

    # 转为稀疏格式（node-feature 映射）
    feat_node_val = defaultdict(list)
    for (node_id, feat_id), val in zip(nonzero_idx.tolist(), nonzero_val.tolist()):
        feat_node_val[feat_id].append((node_id, val))

    del nonzero_idx, nonzero_val

    feature_scores = {}
    sparse_acts = sparse_acts.to(device)

    for feat_id, _ in tqdm(feat_node_val.items(), desc="Region-level contrastive search"):
        feature_acts = sparse_acts.select(1, feat_id).to_dense()
        pos_vals = feature_acts[pos_all_nodes]
        neg_vals = feature_acts[neg_all_nodes]

        # activation density
        pos_density = scatter_mean((pos_vals > 0).float(), pos_region_ids, dim=0)
        neg_density = scatter_mean((neg_vals > 0).float(), neg_region_ids, dim=0)

        # non-zero mean
        pos_nz_sum = scatter_sum(pos_vals, pos_region_ids, dim=0)
        pos_nz_count = scatter_sum((pos_vals > 0).float(), pos_region_ids, dim=0)
        pos_nz_mean = (pos_nz_sum / (pos_nz_count + 1e-6)).cpu().numpy()

        neg_nz_sum = scatter_sum(neg_vals, neg_region_ids, dim=0)
        neg_nz_count = scatter_sum((neg_vals > 0).float(), neg_region_ids, dim=0)
        neg_nz_mean = (neg_nz_sum / (neg_nz_count + 1e-6)).cpu().numpy()
        
        # mean
        pos_means = scatter_mean(pos_vals, pos_region_ids, dim=0).cpu().numpy()
        neg_means = scatter_mean(neg_vals, neg_region_ids, dim=0).cpu().numpy()

        std_all = np.std(np.concatenate([pos_means, neg_means]))
        delta = (np.mean(pos_means) - np.mean(neg_means)) / (std_all + 1e-6)
        stat, pval = mannwhitneyu(pos_means, neg_means, alternative="greater")

        feature_scores[feat_id] = {
            "delta_mean": delta,
            "pval": pval,
            "pos_density": pos_density.mean().item(),
            "neg_density": neg_density.mean().item(),
            "pos_nonzero_mean": np.mean(pos_nz_mean),
            "neg_nonzero_mean": np.mean(neg_nz_mean)
        }

    # 排序与输出
    significant = {k: v for k, v in feature_scores.items() if v["pval"] < 0.05}
    top_feats = dict(sorted(significant.items(), key=lambda x: -x[1]["delta_mean"])[:50])
    return top_feats


def cfs_node(activation_path, annotator):
    device = utils.get_available_gpu()
    data = torch.load(activation_path, weights_only=True)
    sparse_acts = data["sparse_acts"].coalesce()
    prot_lens = data["prot_lens"]
    nonzero_idx = sparse_acts.indices().T  # [nnz，2]
    nonzero_val = sparse_acts.values() 


    domain_mask = annotator.get_coarse_domain_matrix(prot_lens).sum(dim=1).bool().to(device)
    domain_nodes = set(torch.nonzero(domain_mask).view(-1).tolist())
    non_domain_nodes = set(torch.nonzero(~domain_mask).view(-1).tolist())

    del data, sparse_acts, domain_mask

    # 转为稀疏格式（node-feature 映射）
    feat_node_val = defaultdict(lambda: {'pos': [], 'neg': []})
    for (node_id, feat_id), val in zip(nonzero_idx.tolist(), nonzero_val.tolist()):
        if node_id in domain_nodes:
            feat_node_val[feat_id]['pos'].append(val)
        elif node_id in non_domain_nodes:
            feat_node_val[feat_id]['neg'].append(val)

    del nonzero_idx, nonzero_val

    feature_scores = {}

    for feat_id, group in tqdm(feat_node_val.items(), desc="Node-level contrastive search"):
        if len(group['pos']) < 10 or len(group['neg']) < 10:
            continue

        pos_vals = torch.tensor(group['pos'], device=device)
        neg_vals = torch.tensor(group['neg'], device=device)

        # density (非零比例)
        pos_density = (pos_vals > 0).float().mean().item()
        neg_density = (neg_vals > 0).float().mean().item()

        # 非零激活均值
        pos_nz_vals = pos_vals[pos_vals > 0]
        neg_nz_vals = neg_vals[neg_vals > 0]
        pos_nz_mean = pos_nz_vals.mean().item() if len(pos_nz_vals) > 0 else 0.0
        neg_nz_mean = neg_nz_vals.mean().item() if len(neg_nz_vals) > 0 else 0.0

        # 均值对比
        pos_mean = pos_vals.mean().item()
        neg_mean = neg_vals.mean().item()
        all_vals_np = torch.cat([pos_vals, neg_vals]).detach().cpu().numpy()
        std_all = np.std(all_vals_np)
        delta = (pos_mean - neg_mean) / (std_all + 1e-6)

        # 非参数检验
        try:
            stat, pval = mannwhitneyu(
                pos_vals.detach().cpu().numpy(),
                neg_vals.detach().cpu().numpy(),
                alternative="greater"
            )
        except ValueError:
            pval = 1.0  # 样本不足时 fallback

        feature_scores[feat_id] = {
            "delta_mean": delta,
            "pval": pval,
            "pos_density": pos_density,
            "neg_density": neg_density,
            "pos_nonzero_mean": pos_nz_mean,
            "neg_nonzero_mean": neg_nz_mean,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
        }

    # 排序与输出
    significant = {k: v for k, v in feature_scores.items() if v["pval"] < 0.05}
    top_feats = dict(sorted(significant.items(), key=lambda x: -x[1]["delta_mean"])[:50])
    return top_feats


def plot_feature_tracks(
        prot_id: str,
        annotator,
        top_k_feats = [2170, 6857, 1592, 1637, 1428],
        target_domains = None,
        act_path = "/mnt/disk1/nwn/code/sae/test_activations_raw.pt"):

    # ── 1. 读取稀疏激活 及 元信息 ─────────────────────────
    data         = torch.load(act_path, weights_only=True, map_location="cpu")
    sp           = data["sparse_acts"].coalesce()          # sparse COO [N, D]
    prot_lens    = data["prot_lens"]                       # {uid: length}
    offsets      = data["offsets"]                         # {uid: (start, end)}

    start_idx, end_idx = offsets[prot_id]
    seq_len             = end_idx - start_idx
    k                   = len(top_k_feats)

    # ── 2. 在稀疏索引中筛选「本蛋白 & top‑k 特征列」────────
    rows, cols = sp.indices()
    vals       = sp.values()

    row_mask = (rows >= start_idx) & (rows < end_idx)
    col_mask = torch.isin(cols, torch.tensor(top_k_feats))
    sel_mask = row_mask & col_mask

    rows_sel = rows[sel_mask] - start_idx                # local row idx
    cols_sel = cols[sel_mask]
    vals_sel = vals[sel_mask]

    # col_id → 0…k‑1 映射
    feat2col = {fid: i for i, fid in enumerate(top_k_feats)}
    col_ids  = torch.tensor([feat2col[int(c)] for c in cols_sel], dtype=torch.long)

    # ── 3. 构建小稠密矩阵 [k, seq_len] ───────────────────────
    acts = torch.zeros(k, seq_len, dtype=vals_sel.dtype)
    acts.index_put_((col_ids, rows_sel), vals_sel, accumulate=True)  # 若同位置有多值可累加

    # ── 4. 构造结构域掩码 ───────────────────────────────────
    if target_domains is None:
        mask = annotator.get_coarse_domain_matrix({prot_id: seq_len})
    else:
        mask = annotator.get_specific_domain_matrix({prot_id: seq_len}, target_domains).sum(-1, keepdim=True)
    domain_mask = mask.bool().T.numpy()      # shape [1, seq_len]

    fig, axs = plt.subplots(
        k + 1, 1, sharex=True,
        figsize=(12, 0.8 * k + 2),
        gridspec_kw={'height_ratios': [1]*k + [0.3]}
    )

    for i, fid in enumerate(top_k_feats):
        # 灰色背景
        domain_region = (domain_mask[0] == 1)
        in_region = False
        region_start = 0
        for pos, val in enumerate(domain_region):
            if val and not in_region:
                region_start = pos
                in_region = True
            elif not val and in_region:
                axs[i].axvspan(region_start, pos, color="lightgray", alpha=0.5, zorder=0)
                in_region = False
        if in_region:
            axs[i].axvspan(region_start, seq_len, color="lightgray", alpha=0.5, zorder=0)

        # 曲线
        axs[i].fill_between(range(seq_len), 0, acts[i], step="mid", color="steelblue")

        # 实际最小值和最大值（保留原值）
        ymin = acts[i].min().item()
        ymax = acts[i].max().item()

        # 刻度线
        axs[i].spines[["top", "right"]].set_visible(False)

        ymin_str = f"{ymin:.1e}"
        ymax_str = f"{ymax:.1e}"

        axs[i].set_ylim(ymin_str, ymax_str)
        # axs[i].set_yticklabels([ymin_str, ymax_str], fontsize=9)

        # 标签放在数值左边
        axs[i].set_ylabel(f"f/{fid}", rotation=0, labelpad=35, fontsize=9)

    # 最后一行画结构域
    axs[-1].imshow(domain_mask, aspect="auto", cmap="Reds", extent=[0, seq_len, 0, 1])
    axs[-1].set_ylabel("Domain", rotation=0, labelpad=35, fontsize=9)
    axs[-1].set_yticks([])
    axs[-1].spines[["top", "right"]].set_visible(False)
    axs[-1].set_xlabel(f"Residue Position • {prot_id}")

    plt.tight_layout()
    plt.show()



def extract_pfam_acc_sets_by_type(domain_txt_path):
    with open(domain_txt_path, 'r') as f:
        lines = f.readlines()

    pfam_acc_sets = defaultdict(lambda: {'core': set(), 'auxiliary': set()})

    current_category = None
    current_type = None  # 'core' or 'auxiliary'

    for line in lines:
        line = line.strip()
        if not line or line.startswith('Accession'):
            continue
        if line.startswith('#'):
            section = line[1:].strip().lower()
            if section in ['core', 'auxiliary']:
                current_type = section
            else:
                current_category = section
                current_type = None  # reset until explicitly seen
        elif current_category and current_type:
            parts = line.split('\t')
            if parts:
                pfam_acc = parts[0]
                pfam_acc_sets[current_category][current_type].add(pfam_acc)

    return pfam_acc_sets


def run_family(target_domains):
    annot = NodeDomainAnnotator()
    activation_path = "/mnt/disk1/nwn/code/sae/test_activations_sparse.pt"
    return cfs_region(activation_path, annot, target_domains)


if __name__ == "__main__":
    device = utils.get_available_gpu()
    annotator = NodeDomainAnnotator()
    # annotator.save_ids()
    activation_path = "/mnt/disk1/nwn/code/sae/test_activations_sparse.pt"
    # top_feats = cfs_node(activation_path, annotator) # 不使用

    domain_file = "/mnt/disk1/nwn/data/domain/all.txt"  
    acc_sets = extract_pfam_acc_sets_by_type(domain_file)
    results = {}
    for category, groups in acc_sets.items():
        # pf = sorted(groups['core']) + sorted(groups['auxiliary'])
        target_domains = sorted(acc_sets[category]['core']) + sorted(acc_sets[category]['auxiliary'])
        top_feats = cfs_region(activation_path, annotator, target_domains, device=device)
        top5 = dict(list(top_feats.items())[:5])
        results[category] = top5
        print(f"{category.upper()}:")
        for i, (key, value) in enumerate(top5.items()):
            print(f"{key}: {value}")
        del top_feats
        torch.cuda.empty_cache()          # 清空未引用张量
        torch.cuda.ipc_collect()          # 进一步回收显存碎片

    # 展开为表格保存
    df = pd.DataFrame([
        {"category": cat, "feature": k, **v}
        for cat, top5 in results.items()
        for k, v in top5.items()
    ])
    df.to_csv("/mnt/disk1/nwn/code/sae/feat_ids.csv", index=False)

'''
    # prot_id = "P57723" # KH domain protein
    prot_id = "P31943" # RRM
    # prot_id = "P46531"  # 示例蛋白质ID
    plot_feature_tracks(prot_id, annotator,
                        top_k_feats = list(top_feats.keys())[:5],
                        target_domains=target_domains
                        )
'''