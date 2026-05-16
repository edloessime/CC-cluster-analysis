import io
import warnings
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

PALETTE = 'tab10'

plt.rcParams.update({
    'figure.dpi': 110,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Customer Segmentation',
    layout='wide',
)

# ── Session state defaults ─────────────────────────────────────────────────────

_DEFAULTS = {
    'raw_df': None,
    'col_roles': {},
    'clean_df': None,
    'feature_matrix': None,
    'feature_columns': None,
    'scaler': None,
    'optimal_k': None,
    'cluster_labels': None,
    'km_model': None,
    'clean_log': None,
    'k_results': None,
    'cluster_metrics': None,
    'run_history': [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Data helpers ───────────────────────────────────────────────────────────────

def classify_column(series):
    name_lower = series.name.lower()
    nunique = series.nunique(dropna=True)
    n = len(series)

    if any(tok in name_lower for tok in ['id', 'uuid', 'key', 'index']) or nunique == n:
        return 'id'
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    if series.dtype == object:
        try:
            pd.to_datetime(series.dropna().head(20), infer_datetime_format=True)
            return 'datetime'
        except Exception:
            pass
    if pd.api.types.is_numeric_dtype(series):
        return 'binary' if nunique <= 2 else 'numeric'
    if series.dtype == object:
        return 'categorical' if nunique / n < 0.5 else 'high_cardinality_text'
    return 'unknown'


def evaluate_data(df):
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        rows.append({
            'Column': col,
            'Dtype': str(s.dtype),
            'Role': classify_column(s),
            'Unique': int(s.nunique(dropna=True)),
            'Missing': missing,
            'Missing %': f'{missing / len(s) * 100:.1f}%',
            'Sample Values': str(list(s.dropna().unique()[:3])),
        })
    report = pd.DataFrame(rows)
    st.session_state['col_roles'] = {r['Column']: r['Role'] for r in rows}
    return report


def clean_and_preprocess(df, col_roles):
    log = []

    drop_roles = {'id', 'datetime', 'high_cardinality_text', 'unknown'}
    drop_cols = list(
        {c for c, r in col_roles.items() if r in drop_roles}
        | {c for c in df.columns if df[c].isna().mean() > 0.5}
    )
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.append(f'Dropped {len(drop_cols)} column(s): {drop_cols}')

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        log.append(f'Removed {removed:,} duplicate row(s)')

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    if numeric_cols:
        df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
        log.append(f'Imputed {len(numeric_cols)} numeric column(s) with median')
    if cat_cols:
        for col in cat_cols:
            if df[col].isna().any():
                mode = df[col].mode()
                df[col] = df[col].fillna(mode[0] if not mode.empty else 'Unknown')
        log.append(f'Imputed {len(cat_cols)} categorical column(s) with mode')
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
        log.append(f'One-hot encoded {len(cat_cols)} categorical column(s)')

    feature_cols = df.columns.tolist()
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    log.append(f'Standard-scaled {len(feature_cols)} feature(s)')

    return df, X, feature_cols, scaler, log


def find_optimal_k(X, k_min, k_max):
    inertias, silhouettes, db_scores = [], [], []
    k_range = list(range(k_min, k_max + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
    best_k = k_range[int(np.argmax(silhouettes))]
    return k_range, inertias, silhouettes, db_scores, best_k


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _cmap(k):
    return plt.cm.get_cmap(PALETTE, k)


def fig_k_selection(k_range, inertias, silhouettes, db_scores, best_k):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        (inertias,   'Elbow Method (Inertia)',              'steelblue',   'Inertia'),
        (silhouettes,'Silhouette Score (higher = better)',   'seagreen',    'Score'),
        (db_scores,  'Davies-Bouldin (lower = better)',      'darkorange',  'Score'),
    ]
    for ax, (vals, title, color, ylabel) in zip(axes, specs):
        ax.plot(k_range, vals, marker='o', color=color, linewidth=2)
        ax.axvline(best_k, color='red', linestyle='--', alpha=0.7, label=f'K={best_k}')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('K')
        ax.set_ylabel(ylabel)
        ax.legend()
    plt.tight_layout()
    return fig


def fig_pca_scatter(X, labels, k):
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X)
    var = pca.explained_variance_ratio_
    cmap = _cmap(k)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(k):
        mask = labels == c
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=35, alpha=0.6,
                   color=cmap(c), label=f'Cluster {c}')
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.1f}% variance)')
    ax.set_title('Customer Segments — PCA 2-D Projection', fontweight='bold')
    ax.legend(title='Cluster')
    plt.tight_layout()
    return fig


def fig_cluster_sizes(labels, k):
    unique, counts = np.unique(labels, return_counts=True)
    cmap = _cmap(k)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f'Cluster {c}' for c in unique], counts,
                  color=[cmap(i) for i in range(k)], edgecolor='white')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f'{cnt:,}\n({cnt/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    ax.set_title('Cluster Size Distribution', fontweight='bold')
    ax.set_ylabel('Number of Customers')
    ax.set_ylim(0, max(counts) * 1.22)
    plt.tight_layout()
    return fig


def fig_feature_heatmap(clean_df, feature_cols, labels, k, top_n=15):
    df_tmp = clean_df[feature_cols].copy()
    df_tmp['Cluster'] = labels
    means = df_tmp.groupby('Cluster')[feature_cols].mean()
    top_features = means.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
    hm = means[top_features]
    hm_z = (hm - hm.mean()) / (hm.std() + 1e-9)

    # Shorten column labels: replace underscores with spaces and wrap at 14 chars
    def short_label(name):
        name = name.replace('_', ' ')
        if len(name) > 14:
            words = name.split()
            lines, line = [], []
            for w in words:
                if sum(len(x) + 1 for x in line) + len(w) > 14:
                    lines.append(' '.join(line))
                    line = [w]
                else:
                    line.append(w)
            if line:
                lines.append(' '.join(line))
            return '\n'.join(lines)
        return name

    display_labels = [short_label(f) for f in top_features]
    hm_z.columns = display_labels
    hm_z.index = [f'Cluster {i}' for i in hm_z.index]

    cell_w, cell_h = 1.1, 0.7
    fig_w = max(10, len(top_features) * cell_w + 2)
    fig_h = max(4, k * cell_h + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        hm_z,
        annot=True, fmt='.2f', annot_kws={'size': 9},
        cmap='RdYlGn', linewidths=0.5, linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Z-score', 'shrink': 0.8},
    )
    ax.set_title(
        f'Top {top_n} Differentiating Features per Cluster (Z-scored means)',
        fontweight='bold', fontsize=12, pad=12,
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(axis='x', labelsize=8, rotation=35)
    ax.tick_params(axis='y', labelsize=9, rotation=0)
    fig.subplots_adjust(bottom=0.22, left=0.12, right=0.95, top=0.9)
    return fig, top_features


def fig_feature_boxplots(clean_df, feature_cols, labels, top_features, k, n_plots=6):
    selected = top_features[:n_plots]
    ncols = 3
    nrows = max(1, (len(selected) + ncols - 1) // ncols)
    cmap = _cmap(k)
    df_tmp = clean_df[feature_cols].copy()
    df_tmp['Cluster'] = labels
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
    axes = np.array(axes).flatten()
    for ax, feat in zip(axes, selected):
        data = [df_tmp[df_tmp['Cluster'] == c][feat].values.astype(float) for c in range(k)]
        bp = ax.boxplot(data, patch_artist=True)
        for patch, i in zip(bp['boxes'], range(k)):
            patch.set_facecolor(cmap(i))
            patch.set_alpha(0.7)
        ax.set_xticklabels([f'C{c}' for c in range(k)])
        ax.set_title(feat, fontsize=9, fontweight='bold')
    for ax in axes[len(selected):]:
        ax.set_visible(False)
    fig.suptitle('Feature Distributions per Cluster (Top Differentiators)', fontweight='bold')
    plt.tight_layout()
    return fig


def fig_silhouette(X, labels, k):
    scores = silhouette_samples(X, labels)
    cmap = _cmap(k)
    fig, ax = plt.subplots(figsize=(8, 4))
    y_lo = 10
    for c in range(k):
        cs = np.sort(scores[labels == c])
        y_hi = y_lo + len(cs)
        ax.fill_betweenx(np.arange(y_lo, y_hi), 0, cs,
                         facecolor=cmap(c), alpha=0.7, label=f'Cluster {c}')
        ax.text(-0.05, y_lo + len(cs) / 2, str(c), fontsize=9)
        y_lo = y_hi + 10
    avg = silhouette_score(X, labels)
    ax.axvline(avg, color='red', linestyle='--', label=f'Avg = {avg:.3f}')
    ax.set_title('Silhouette Plot — Per-Sample Scores by Cluster', fontweight='bold')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_yticks([])
    ax.legend(loc='lower right')
    plt.tight_layout()
    return fig


# ── Cluster profile helpers ────────────────────────────────────────────────────

def build_cluster_profiles(clean_df, feature_cols, labels, k, top_n=5):
    """Return a list of dicts describing each cluster in plain terms."""
    df_tmp = clean_df[feature_cols].copy()
    df_tmp['Cluster'] = labels
    means = df_tmp.groupby('Cluster')[feature_cols].mean()
    hm_z  = (means - means.mean()) / (means.std() + 1e-9)
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)

    profiles = []
    for c in range(k):
        z = hm_z.loc[c]
        high = [(col.replace('_', ' '), round(val, 2))
                for col, val in z.nlargest(top_n).items() if val > 0.3]
        low  = [(col.replace('_', ' '), round(val, 2))
                for col, val in z.nsmallest(top_n).items() if val < -0.3]
        profiles.append({
            'cluster': c,
            'count':   int(counts[c]),
            'pct':     round(counts[c] / total * 100, 1),
            'high':    high,
            'low':     low,
        })
    return profiles


# ── PDF report builder ─────────────────────────────────────────────────────────

def _pdf_cover(filename, n_rows, feature_cols, k, inertia, sil, db, profiles):
    """Return a matplotlib figure that serves as the PDF cover/summary page."""
    unique_counts = [(p['cluster'], p['count'], p['pct']) for p in profiles]
    cmap = plt.cm.get_cmap(PALETTE, k)

    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Header bar
    ax.add_patch(plt.Rectangle((0, 0.88), 1, 0.12,
                                transform=ax.transAxes, color='#2c3e50', zorder=0))
    ax.text(0.5, 0.945, 'Customer Segmentation Report',
            transform=ax.transAxes, fontsize=22, fontweight='bold',
            color='white', ha='center', va='center')

    y = 0.82
    ax.text(0.5, y, f'File: {filename}',
            transform=ax.transAxes, fontsize=11, ha='center', color='#555')
    y -= 0.04
    ax.text(0.5, y, pd.Timestamp.now().strftime('%B %d, %Y'),
            transform=ax.transAxes, fontsize=10, ha='center', color='#888')

    # Metrics row
    y -= 0.07
    for i, (label, val) in enumerate([
        ('Clusters (K)', str(k)),
        ('Rows', f'{n_rows:,}'),
        ('Features', str(len(feature_cols))),
        ('Silhouette', f'{sil:.4f}'),
        ('Davies-Bouldin', f'{db:.4f}'),
        ('Inertia', f'{inertia:,.0f}'),
    ]):
        x = 0.08 + i * 0.155
        ax.add_patch(plt.Rectangle((x, y - 0.055), 0.14, 0.065,
                                    transform=ax.transAxes,
                                    color='#ecf0f1', zorder=1, clip_on=False))
        ax.text(x + 0.07, y - 0.01, val, transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='center', va='center', color='#2c3e50')
        ax.text(x + 0.07, y - 0.042, label, transform=ax.transAxes,
                fontsize=7, ha='center', va='center', color='#888')

    # Cluster size bars
    y -= 0.12
    ax.text(0.5, y, 'Cluster Size Overview',
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            ha='center', color='#2c3e50')
    y -= 0.04
    bar_w = 0.6 / k
    for c, count, pct in unique_counts:
        x = 0.2 + c * bar_w
        bar_h = 0.08 * (pct / max(p['pct'] for p in profiles))
        ax.add_patch(plt.Rectangle((x, y - bar_h), bar_w * 0.85, bar_h,
                                    transform=ax.transAxes,
                                    color=cmap(c), alpha=0.85, clip_on=False))
        ax.text(x + bar_w * 0.425, y - bar_h - 0.018,
                f'C{c}\n{count:,}\n({pct}%)',
                transform=ax.transAxes, fontsize=8, ha='center', va='top', color='#333')

    # Cluster profiles text
    y -= 0.16
    ax.text(0.5, y, 'Cluster Profiles',
            transform=ax.transAxes, fontsize=13, fontweight='bold',
            ha='center', color='#2c3e50')
    y -= 0.03

    for p in profiles:
        ax.add_patch(plt.Rectangle((0.05, y - 0.095), 0.9, 0.09,
                                    transform=ax.transAxes,
                                    color=cmap(p['cluster']), alpha=0.12, clip_on=False))
        ax.add_patch(plt.Rectangle((0.05, y - 0.095), 0.008, 0.09,
                                    transform=ax.transAxes,
                                    color=cmap(p['cluster']), alpha=0.9, clip_on=False))
        ax.text(0.07, y - 0.015,
                f"Cluster {p['cluster']}  —  {p['count']:,} customers ({p['pct']}%)",
                transform=ax.transAxes, fontsize=9, fontweight='bold', color='#2c3e50')
        high_txt = '▲ ' + ',  '.join(f[0] for f in p['high']) if p['high'] else '—'
        low_txt  = '▼ ' + ',  '.join(f[0] for f in p['low'])  if p['low']  else '—'
        ax.text(0.07, y - 0.042, f'High: {high_txt}',
                transform=ax.transAxes, fontsize=8, color='#27ae60')
        ax.text(0.07, y - 0.068, f'Low:  {low_txt}',
                transform=ax.transAxes, fontsize=8, color='#e74c3c')
        y -= 0.11

    plt.tight_layout()
    return fig


def build_pdf_report(filename, n_rows, feature_cols, labels, k, km, sil, db,
                     clean_df, feature_matrix, profiles):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 — cover + summary
        fig = _pdf_cover(filename, n_rows, feature_cols, k,
                         km.inertia_, sil, db, profiles)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2 — PCA scatter
        fig = fig_pca_scatter(feature_matrix, labels, k)
        fig.suptitle('PCA 2-D Projection', fontsize=13, fontweight='bold', y=1.01)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3 — cluster sizes
        fig = fig_cluster_sizes(labels, k)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 4 — feature heatmap
        fig, _ = fig_feature_heatmap(clean_df, feature_cols, labels, k, top_n=15)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 5 — silhouette
        fig = fig_silhouette(feature_matrix, labels, k)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        pdf.infodict().update({
            'Title': 'Customer Segmentation Report',
            'Subject': f'{filename} — K={k}',
        })

    buf.seek(0)
    return buf.read()


# ── History helpers ────────────────────────────────────────────────────────────

def make_history_record(filename, n_rows, feature_cols, k, km, labels, sil, db):
    unique, counts = np.unique(labels, return_counts=True)
    sizes = ', '.join(f'C{c}:{n}' for c, n in zip(unique, counts))
    return {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'File': filename,
        'Rows': n_rows,
        'Features': len(feature_cols),
        'K': int(k),
        'Inertia': round(float(km.inertia_), 1),
        'Silhouette': round(float(sil), 4),
        'Davies-Bouldin': round(float(db), 4),
        'Cluster Sizes': sizes,
        'Notes': '',
    }


_HISTORY_COLS = {'Timestamp', 'File', 'K', 'Silhouette', 'Davies-Bouldin'}


def fig_history_comparison(history, filename):
    runs = [r for r in history if r['File'] == filename]
    if len(runs) < 2:
        return None
    df = pd.DataFrame(runs).sort_values('K')
    best_k = df.loc[df['Silhouette'].idxmax(), 'K']

    fig, ax1 = plt.subplots(figsize=(7, 3))
    ax2 = ax1.twinx()
    ax1.plot(df['K'], df['Silhouette'], marker='o', color='seagreen',
             linewidth=2, label='Silhouette')
    ax2.plot(df['K'], df['Inertia'], marker='s', color='steelblue',
             linewidth=2, linestyle='--', label='Inertia')
    ax1.axvline(best_k, color='red', linestyle=':', alpha=0.8,
                label=f'Best K={best_k}')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Silhouette Score', color='seagreen')
    ax2.set_ylabel('Inertia', color='steelblue')
    ax1.set_title(f'K Comparison — {filename}', fontweight='bold', fontsize=10)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper right')
    plt.tight_layout()
    return fig


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title('Customer Segmentation Analysis')
st.caption('Upload a CSV or Excel file to profile, clean, cluster, and visualize your customer data.')

# ── Sidebar — Run History ──────────────────────────────────────────────────────

with st.sidebar:
    st.header('Run History')

    restore_file = st.file_uploader(
        'Restore a saved history.csv',
        type=['csv'],
        key='history_uploader',
    )
    if restore_file is not None and restore_file.name != st.session_state.get('history_filename'):
        try:
            restored = pd.read_csv(restore_file)
            if _HISTORY_COLS.issubset(set(restored.columns)):
                restored['Notes'] = restored.get('Notes', '').fillna('')
                st.session_state['run_history'] = restored.to_dict('records')
                st.session_state['history_filename'] = restore_file.name
                st.success(f'Restored {len(restored)} run(s).')
            else:
                st.error('This does not look like a history file — expected columns are missing.')
        except Exception as e:
            st.error(f'Could not read file: {e}')

    history = st.session_state['run_history']

    if not history:
        st.info('No runs yet. Complete a clustering run to start recording history.')
    else:
        hdf = pd.DataFrame(history)

        st.download_button(
            label='Download history.csv',
            data=hdf.to_csv(index=False).encode(),
            file_name='history.csv',
            mime='text/csv',
        )

        st.dataframe(
            hdf[['Timestamp', 'File', 'K', 'Silhouette', 'Davies-Bouldin']],
            width='stretch',
        )

        best = max(history, key=lambda r: r['Silhouette'])
        st.success(
            f"Best overall: **{best['File']}** "
            f"K={best['K']} — Silhouette {best['Silhouette']:.4f}"
        )

        current_file = st.session_state.get('uploaded_filename')
        if current_file:
            comparison_fig = fig_history_comparison(history, current_file)
            if comparison_fig:
                st.subheader(f'K Comparison')
                st.pyplot(comparison_fig)
                file_runs = [r for r in history if r['File'] == current_file]
                best_for_file = max(file_runs, key=lambda r: r['Silhouette'])
                st.info(
                    f"Best for **{current_file}**: "
                    f"K={best_for_file['K']} — "
                    f"Silhouette {best_for_file['Silhouette']:.4f}"
                )

# ── 1. Upload ──────────────────────────────────────────────────────────────────

st.header('1. Upload Data')

if st.session_state['raw_df'] is None:
    uploaded = st.file_uploader('Choose a CSV or Excel file', type=['csv', 'xlsx', 'xls'])
    if uploaded is not None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state['raw_df'] = df
            st.session_state['uploaded_filename'] = uploaded.name
            st.rerun()
        except Exception as e:
            st.error(f'Could not read file: {e}')
            st.stop()
    else:
        st.info('Upload a file above to begin.')
        st.stop()
else:
    st.success(
        f"Loaded **{st.session_state['uploaded_filename']}** — "
        f"{st.session_state['raw_df'].shape[0]:,} rows × "
        f"{st.session_state['raw_df'].shape[1]} columns"
    )
    if st.button('Upload a different file'):
        for key in _DEFAULTS:
            st.session_state[key] = _DEFAULTS[key]
        st.session_state['uploaded_filename'] = None
        st.rerun()

raw_df = st.session_state['raw_df']

# ── 2. Evaluate ────────────────────────────────────────────────────────────────

st.header('2. Data Evaluation')
report_df = evaluate_data(raw_df)

c1, c2, c3 = st.columns(3)
c1.metric('Rows', f'{raw_df.shape[0]:,}')
c2.metric('Columns', raw_df.shape[1])
c3.metric('Duplicate Rows', f'{raw_df.duplicated().sum():,}')

st.dataframe(report_df, width='stretch')

high_missing = report_df[report_df['Missing'].astype(int) / len(raw_df) > 0.5]
if not high_missing.empty:
    st.warning(f"Columns with >50% missing values (will be dropped): {list(high_missing['Column'])}")

numeric_cols_eval = report_df[report_df['Role'] == 'numeric']['Column'].tolist()
if numeric_cols_eval:
    with st.expander('Numeric column statistics'):
        st.dataframe(raw_df[numeric_cols_eval].describe().round(2), width='stretch')

# ── 3. Clean ───────────────────────────────────────────────────────────────────

st.header('3. Data Cleaning & Preprocessing')

if st.button('Run Cleaning & Preprocessing', type='primary'):
    with st.spinner('Cleaning data...'):
        clean_df, X, feature_cols, scaler, log = clean_and_preprocess(
            raw_df.copy(), st.session_state['col_roles']
        )
    st.session_state.update({
        'clean_df': clean_df, 'feature_matrix': X,
        'feature_columns': feature_cols, 'scaler': scaler,
        'cluster_labels': None, 'optimal_k': None,
        'clean_log': log, 'k_results': None, 'cluster_metrics': None,
    })

if st.session_state['clean_log']:
    for msg in st.session_state['clean_log']:
        st.write(f'- {msg}')
    st.success(
        f"Ready: {len(st.session_state['clean_df']):,} rows "
        f"× {len(st.session_state['feature_columns'])} features"
    )

if st.session_state['feature_matrix'] is None:
    st.stop()

# ── 4. Find Optimal K ──────────────────────────────────────────────────────────

st.header('4. Find Optimal K')

col_kmin, col_kmax = st.columns(2)
k_min = col_kmin.number_input('K min', min_value=2, max_value=20, value=2)
k_max = col_kmax.number_input('K max', min_value=3, max_value=20, value=10)

if st.button('Evaluate K Range', type='primary'):
    if int(k_min) >= int(k_max):
        st.error('K max must be greater than K min.')
    else:
        with st.spinner(f'Testing K = {int(k_min)} to {int(k_max)}...'):
            k_range, inertias, silhouettes, db_scores, best_k = find_optimal_k(
                st.session_state['feature_matrix'], int(k_min), int(k_max)
            )
        st.session_state['optimal_k'] = best_k
        st.session_state['k_results'] = (k_range, inertias, silhouettes, db_scores, best_k)
        st.session_state['cluster_metrics'] = None

if st.session_state['k_results']:
    k_range, inertias, silhouettes, db_scores, best_k = st.session_state['k_results']
    st.pyplot(fig_k_selection(k_range, inertias, silhouettes, db_scores, best_k))
    st.info(f'Suggested K = **{best_k}** (highest silhouette score). '
            'You can override this below.')

# ── 5. Clustering ──────────────────────────────────────────────────────────────

st.header('5. K-Means Clustering')

default_k = st.session_state['optimal_k'] if st.session_state['optimal_k'] else 3
chosen_k = st.number_input(
    'Number of clusters (K) — adjust to override the suggestion above',
    min_value=2, max_value=30, value=default_k,
)

if st.button('Run Clustering', type='primary'):
    with st.spinner(f'Fitting K-Means with K={int(chosen_k)}...'):
        km = KMeans(n_clusters=int(chosen_k), random_state=42, n_init='auto', max_iter=500)
        labels = km.fit_predict(st.session_state['feature_matrix'])

    cdf = st.session_state['clean_df'].copy()
    cdf['Cluster'] = labels
    sil = silhouette_score(st.session_state['feature_matrix'], labels)
    db  = davies_bouldin_score(st.session_state['feature_matrix'], labels)
    unique, counts = np.unique(labels, return_counts=True)
    st.session_state.update({
        'cluster_labels': labels,
        'km_model': km,
        'clean_df': cdf,
        'optimal_k': int(chosen_k),
        'cluster_metrics': {
            'inertia': km.inertia_, 'sil': sil, 'db': db,
            'unique': unique, 'counts': counts,
        },
    })

    # ── Save to history ──────────────────────────────────────────────────────
    fname = st.session_state.get('uploaded_filename', 'unknown')
    existing = st.session_state['run_history']
    duplicate = next(
        (i for i, r in enumerate(existing)
         if r['File'] == fname and r['K'] == int(chosen_k)),
        None,
    )
    record = make_history_record(
        fname, len(raw_df), st.session_state['feature_columns'],
        int(chosen_k), km, labels, sil, db,
    )
    if duplicate is not None:
        existing[duplicate] = record
    else:
        existing.append(record)
    st.rerun()

if st.session_state['cluster_metrics']:
    m = st.session_state['cluster_metrics']
    m1, m2, m3 = st.columns(3)
    m1.metric('Inertia', f"{m['inertia']:,.1f}")
    m2.metric('Silhouette Score', f"{m['sil']:.4f}")
    m3.metric('Davies-Bouldin', f"{m['db']:.4f}")
    labels_total = sum(m['counts'])
    st.dataframe(pd.DataFrame({
        'Cluster': m['unique'],
        'Count': m['counts'],
        'Share': [f'{c / labels_total * 100:.1f}%' for c in m['counts']],
    }), width='stretch')
    st.success('Clustering complete. See visualizations below.')

if st.session_state['cluster_labels'] is None:
    st.stop()

# ── 6. Visualizations ──────────────────────────────────────────────────────────

st.header('6. Visualizations')

X      = st.session_state['feature_matrix']
labels = st.session_state['cluster_labels']
k      = st.session_state['optimal_k']
cdf    = st.session_state['clean_df']
fcols  = st.session_state['feature_columns']

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['Cluster Profiles', 'PCA Scatter', 'Cluster Sizes', 'Feature Heatmap', 'Box Plots', 'Silhouette']
)

profiles = build_cluster_profiles(cdf, fcols, labels, k)
cmap = plt.cm.get_cmap(PALETTE, k)

with tab0:
    cols_per_row = min(k, 3)
    rows = [profiles[i:i + cols_per_row] for i in range(0, k, cols_per_row)]
    for row in rows:
        card_cols = st.columns(len(row))
        for col, p in zip(card_cols, row):
            hex_color = '#%02x%02x%02x' % tuple(
                int(v * 255) for v in cmap(p['cluster'])[:3]
            )
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='background:{hex_color};border-radius:6px;"
                        f"padding:6px 10px;margin-bottom:8px'>"
                        f"<span style='color:white;font-size:16px;font-weight:700'>"
                        f"Cluster {p['cluster']}</span>"
                        f"<span style='color:rgba(255,255,255,0.85);font-size:12px;"
                        f"margin-left:10px'>{p['count']:,} customers "
                        f"({p['pct']}%)</span></div>",
                        unsafe_allow_html=True,
                    )
                    if p['high']:
                        st.markdown('**Notably high in:**')
                        for feat, z in p['high']:
                            st.markdown(f"- {feat} &nbsp; `z={z:+.2f}`",
                                        unsafe_allow_html=True)
                    else:
                        st.markdown('*No strongly elevated features*')

                    st.markdown('---')

                    if p['low']:
                        st.markdown('**Notably low in:**')
                        for feat, z in p['low']:
                            st.markdown(f"- {feat} &nbsp; `z={z:+.2f}`",
                                        unsafe_allow_html=True)
                    else:
                        st.markdown('*No strongly suppressed features*')

with tab1:
    st.pyplot(fig_pca_scatter(X, labels, k))

with tab2:
    st.pyplot(fig_cluster_sizes(labels, k))

with tab3:
    top_n = st.slider('Number of top features to show', 5, min(20, len(fcols)), 15)
    heatmap_fig, top_feats = fig_feature_heatmap(cdf, fcols, labels, k, top_n)
    st.pyplot(heatmap_fig)

with tab4:
    _, top_feats = fig_feature_heatmap(cdf, fcols, labels, k, 15)
    n_box = st.slider('Number of features in box plots', 3, min(9, len(fcols)), 6)
    st.pyplot(fig_feature_boxplots(cdf, fcols, labels, top_feats, k, n_box))

with tab5:
    st.pyplot(fig_silhouette(X, labels, k))

# ── 7. Export ──────────────────────────────────────────────────────────────────

st.header('7. Export Segmented Data')

clean_idx = cdf.index
export_df = raw_df.loc[clean_idx].copy()
export_df['Cluster'] = labels

st.dataframe(
    export_df.groupby('Cluster').size().rename('count').reset_index(),
    width='stretch',
)

col_csv, col_pdf = st.columns(2)

with col_csv:
    st.download_button(
        label='Download segmented_customers.csv',
        data=export_df.to_csv(index=False).encode(),
        file_name='segmented_customers.csv',
        mime='text/csv',
        type='primary',
    )

with col_pdf:
    with st.spinner('Building PDF report...'):
        pdf_bytes = build_pdf_report(
            filename=st.session_state['uploaded_filename'],
            n_rows=len(raw_df),
            feature_cols=fcols,
            labels=labels,
            k=k,
            km=st.session_state['km_model'],
            sil=st.session_state['cluster_metrics']['sil'],
            db=st.session_state['cluster_metrics']['db'],
            clean_df=cdf,
            feature_matrix=X,
            profiles=profiles,
        )
    st.download_button(
        label='Download PDF Report',
        data=pdf_bytes,
        file_name='segmentation_report.pdf',
        mime='application/pdf',
        type='primary',
    )
