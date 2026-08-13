import io
import warnings

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
    return plt.colormaps[PALETTE].resampled(k)


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


# ── PDF report builder (reportlab) ────────────────────────────────────────────

def build_pdf_report(filename, n_rows, feature_cols, labels, k, km, sil, db,
                     clean_df, feature_matrix, profiles):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        BaseDocTemplate, Frame, PageTemplate, NextPageTemplate,
        Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    PAGE_W, PAGE_H = letter
    MARGIN = 0.75 * inch
    CW = PAGE_W - 2 * MARGIN          # usable content width
    report_date = pd.Timestamp.now().strftime('%B %d, %Y')

    # Brand palette
    C_NAVY  = colors.HexColor('#1a2744')
    C_BLUE  = colors.HexColor('#2980b9')
    C_LIGHT = colors.HexColor('#f5f7fa')
    C_GREEN = colors.HexColor('#27ae60')
    C_RED   = colors.HexColor('#c0392b')
    C_GRAY  = colors.HexColor('#7f8c8d')
    C_DARK  = colors.HexColor('#2c3e50')

    TAB10 = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
              '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

    def ccolor(c):
        return colors.HexColor(TAB10[c % len(TAB10)])

    def tint(c, alpha=0.12):
        h = TAB10[c % len(TAB10)].lstrip('#')
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return colors.Color((r*alpha + 255*(1-alpha))/255,
                            (g*alpha + 255*(1-alpha))/255,
                            (b*alpha + 255*(1-alpha))/255)

    def ps(name, **kw):
        d = dict(fontName='Helvetica', fontSize=10, textColor=C_DARK, leading=14)
        d.update(kw)
        return ParagraphStyle(name, **d)

    def fig_img(fig, width):
        b = io.BytesIO()
        fig.savefig(b, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        b.seek(0)
        aspect = fig.get_figheight() / fig.get_figwidth()
        img = RLImage(b, width=width, height=width * aspect)
        plt.close(fig)
        return img

    def expand_fig_height(fig, target_h):
        """Expand figure to at least target_h inches tall, redistributing axes."""
        old_w, old_h = fig.get_size_inches()
        if old_h < target_h:
            fig.set_size_inches(old_w, target_h)
            try:
                fig.tight_layout()
            except Exception:
                pass
        return fig

    def pdf_heatmap(top_n=15):
        """Heatmap sized for 7-inch PDF content width (avoids the extreme wide
        aspect ratio of the screen version which collapses to ~2 inches tall)."""
        df_tmp = clean_df[feature_cols].copy()
        df_tmp['Cluster'] = labels
        means = df_tmp.groupby('Cluster')[feature_cols].mean()
        top_feats = means.var(axis=0).sort_values(ascending=False).head(top_n).index.tolist()
        hm = means[top_feats]
        hm_z = (hm - hm.mean()) / (hm.std() + 1e-9)

        def short_label(name):
            name = name.replace('_', ' ')
            if len(name) > 12:
                words = name.split()
                lines, line = [], []
                for w in words:
                    if sum(len(x) + 1 for x in line) + len(w) > 12:
                        lines.append(' '.join(line))
                        line = [w]
                    else:
                        line.append(w)
                if line:
                    lines.append(' '.join(line))
                return '\n'.join(lines)
            return name

        hm_z.columns = [short_label(f) for f in top_feats]
        hm_z.index = [f'Cluster {i}' for i in hm_z.index]
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        sns.heatmap(hm_z, annot=True, fmt='.2f', annot_kws={'size': 8},
                    cmap='RdYlGn', linewidths=0.5, linecolor='white',
                    ax=ax, cbar_kws={'label': 'Z-score', 'shrink': 0.8})
        ax.set_title(f'Top {top_n} Differentiating Features per Cluster (Z-scored means)',
                     fontweight='bold', fontsize=11, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=7, rotation=40)
        ax.tick_params(axis='y', labelsize=9, rotation=0)
        fig.subplots_adjust(bottom=0.30, left=0.12, right=0.95, top=0.90)
        return fig

    # ── Page callbacks ─────────────────────────────────────────────────────────
    def on_cover(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_NAVY)
        canvas.rect(0, PAGE_H - 2.6*inch, PAGE_W, 2.6*inch, fill=1, stroke=0)
        canvas.setFillColor(C_BLUE)
        canvas.rect(0, PAGE_H - 2.65*inch, PAGE_W, 0.05*inch, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 26)
        canvas.drawCentredString(PAGE_W / 2, PAGE_H - 1.4*inch, 'Customer Segmentation Report')
        canvas.restoreState()

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(C_NAVY)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, PAGE_H - 0.52*inch, PAGE_W - MARGIN, PAGE_H - 0.52*inch)
        canvas.setFont('Helvetica-Bold', 8)
        canvas.setFillColor(C_NAVY)
        canvas.drawString(MARGIN, PAGE_H - 0.42*inch, 'Customer Segmentation Report')
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(C_GRAY)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.42*inch, filename)
        canvas.setStrokeColor(colors.HexColor('#dddddd'))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 0.58*inch, PAGE_W - MARGIN, 0.58*inch)
        canvas.setFont('Helvetica', 7)
        canvas.setFillColor(C_GRAY)
        canvas.drawString(MARGIN, 0.42*inch, f'Generated {report_date}')
        canvas.drawCentredString(PAGE_W/2, 0.42*inch, f'Page {doc.page}')
        canvas.drawRightString(PAGE_W - MARGIN, 0.42*inch,
                               f'K={k}  |  Silhouette={sil:.4f}')
        canvas.restoreState()

    # ── Document setup ─────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=letter,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    cover_frame = Frame(MARGIN, MARGIN, CW, PAGE_H - 2*MARGIN, id='cover')
    body_frame  = Frame(MARGIN, 0.75*inch, CW, PAGE_H - MARGIN - 0.85*inch, id='body')
    doc.addPageTemplates([
        PageTemplate(id='Cover', frames=[cover_frame], onPage=on_cover),
        PageTemplate(id='Body',  frames=[body_frame],  onPage=on_page),
    ])

    story = []

    # ── Page 1: Cover ──────────────────────────────────────────────────────────
    # Title is drawn directly on canvas in on_cover; spacer pushes
    # filename/date below the navy header (2.65" from top, margin=0.75").
    story.append(NextPageTemplate('Body'))
    story.append(Spacer(1, 2.1*inch))
    story.append(Paragraph(filename,
                            ps('TF', fontSize=11, textColor=colors.HexColor('#aabdd0'),
                               alignment=TA_CENTER)))
    story.append(Paragraph(report_date,
                            ps('TD', fontSize=10, textColor=colors.HexColor('#aabdd0'),
                               alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*inch))

    # Metric tiles
    metrics = [
        ('Clusters (K)', str(k)), ('Rows', f'{n_rows:,}'),
        ('Features', str(len(feature_cols))), ('Silhouette', f'{sil:.4f}'),
        ('Davies-Bouldin', f'{db:.4f}'), ('Inertia', f'{km.inertia_:,.0f}'),
    ]
    tw = CW / len(metrics)
    mt = Table(
        [[Paragraph(v, ps(f'MV{i}', fontSize=15, fontName='Helvetica-Bold',
                           textColor=C_NAVY, alignment=TA_CENTER))
          for i, (_, v) in enumerate(metrics)],
         [Paragraph(l, ps(f'ML{i}', fontSize=7, textColor=C_GRAY,
                           alignment=TA_CENTER, fontName='Helvetica-Oblique'))
          for i, (l, _) in enumerate(metrics)]],
        colWidths=[tw]*len(metrics),
    )
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_LIGHT),
        ('LINEABOVE', (0,0), (-1,0), 3, C_BLUE),
        ('LINEBEFORE', (1,0), (-1,-1), 0.5, colors.white),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.25*inch))

    # Key findings
    u_arr, c_arr = np.unique(labels, return_counts=True)
    largest_c = int(u_arr[np.argmax(c_arr)])
    sil_note = ('moderate cluster separation'
                if sil > 0.3 else 'weak cluster separation — treat segments as tendencies')
    findings = [
        f'K-Means identified <b>{k} distinct customer segments</b> in this dataset.',
        f'The largest segment is <b>Cluster {largest_c}</b> '
        f'({max(c_arr):,} customers, {max(c_arr)/len(labels)*100:.1f}%).',
        f'A silhouette score of <b>{sil:.4f}</b> indicates {sil_note}.',
    ]
    kf = Table(
        [[Paragraph('Key Findings', ps('KFH', fontName='Helvetica-Bold',
                                        fontSize=10, textColor=C_NAVY))]] +
        [[Paragraph(f'• {f}', ps(f'KF{i}', fontSize=9, textColor=C_DARK, leading=13))]
         for i, f in enumerate(findings)],
        colWidths=[CW],
    )
    kf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eaf4fb')),
        ('LINEABOVE', (0,0), (0,0), 3, C_BLUE),
        ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 10), ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(kf)
    story.append(Spacer(1, 0.25*inch))

    # Cluster summary table
    story.append(Paragraph('Cluster Summary',
                            ps('CS', fontName='Helvetica-Bold', fontSize=11,
                               textColor=C_DARK, spaceAfter=6)))
    hdr_s = ps('TH', fontName='Helvetica-Bold', fontSize=9, textColor=colors.white)
    rows = [[Paragraph(h, hdr_s) for h in
             ['Cluster', 'Customers', 'Share', 'Top Characteristics']]]
    for p in profiles:
        high_txt = ', '.join(f[0] for f in p['high'][:3]) if p['high'] else '—'
        low_txt  = ', '.join(f[0] for f in p['low'][:2])  if p['low']  else '—'
        rows.append([
            Paragraph(f"Cluster {p['cluster']}",
                       ps(f'CN{p["cluster"]}', fontName='Helvetica-Bold',
                          fontSize=9, textColor=colors.white)),
            Paragraph(f"{p['count']:,}",
                       ps(f'CC{p["cluster"]}', fontSize=9, alignment=TA_CENTER)),
            Paragraph(f"{p['pct']}%",
                       ps(f'CP{p["cluster"]}', fontSize=9, alignment=TA_CENTER)),
            Paragraph(
                f'<font color="#27ae60">▲ {high_txt}</font><br/>'
                f'<font color="#c0392b">▼ {low_txt}</font>',
                ps(f'CD{p["cluster"]}', fontSize=8, leading=12)),
        ])
    cws = [CW*0.14, CW*0.14, CW*0.10, CW*0.62]
    cts = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), C_NAVY),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, C_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.25, colors.HexColor('#cccccc')),
        ('ALIGN', (1,0), (2,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 8), ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ])
    for i, p in enumerate(profiles):
        cts.add('BACKGROUND', (0, i+1), (0, i+1), ccolor(p['cluster']))
    ct = Table(rows, colWidths=cws)
    ct.setStyle(cts)
    story.append(ct)

    # ── Page 2: Cluster Profiles ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph('Cluster Profiles',
                            ps('SEC', fontName='Helvetica-Bold', fontSize=16,
                               textColor=C_NAVY, spaceAfter=4)))
    story.append(HRFlowable(width='100%', thickness=2, color=C_NAVY, spaceAfter=8))
    story.append(Paragraph(
        'Each cluster is described by features where its customers score significantly '
        'above (▲) or below (▼) the dataset average. Z-scores indicate magnitude.',
        ps('INTRO', fontSize=9, textColor=C_GRAY, leading=13, spaceAfter=12)))

    for p in profiles:
        cc = ccolor(p['cluster'])
        hdr_tbl = Table(
            [[Paragraph(f"Cluster {p['cluster']}",
                         ps(f'PH{p["cluster"]}', fontName='Helvetica-Bold',
                            fontSize=13, textColor=colors.white)),
              Paragraph(f"{p['count']:,} customers  ({p['pct']}%)",
                         ps(f'PS{p["cluster"]}', fontSize=10,
                            textColor=colors.white, alignment=TA_RIGHT))]],
            colWidths=[CW*0.5, CW*0.5],
        )
        hdr_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), cc),
            ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 12), ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))

        def feat_rows(items, clr):
            return [[Paragraph(f[0], ps(f'FR_{f[0]}', fontSize=9,
                                        textColor=clr, leading=13)),
                     Paragraph(f'z={f[1]:+.2f}',
                                ps(f'FZ_{f[0]}', fontSize=8, textColor=C_GRAY,
                                   alignment=TA_RIGHT))]
                    for f in items] if items else \
                   [[Paragraph('No strongly differentiating features',
                                ps('FRN', fontSize=9, textColor=C_GRAY, fontName='Helvetica-Oblique')),
                     Paragraph('', ps('FZN'))]]

        body_rows = (
            [[Paragraph('▲  Notably High', ps('HL', fontName='Helvetica-Bold',
                                               fontSize=9, textColor=C_GREEN)),
              Paragraph('Z-Score', ps('ZL', fontSize=8, textColor=C_GRAY,
                                      alignment=TA_RIGHT, fontName='Helvetica-Oblique'))]] +
            feat_rows(p['high'], C_GREEN) +
            [[Paragraph('▼  Notably Low', ps('LL', fontName='Helvetica-Bold',
                                              fontSize=9, textColor=C_RED)),
              Paragraph('', ps('LZ'))]] +
            feat_rows(p['low'], C_RED)
        )
        body_tbl = Table(body_rows, colWidths=[CW*0.78, CW*0.22])
        body_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), tint(p['cluster'])),
            ('LINEBELOW', (0,0), (-1,0), 0.5, colors.HexColor('#cccccc')),
            ('LINEBELOW', (0, len(p['high']) or 1), (-1, len(p['high']) or 1),
             0.5, colors.HexColor('#cccccc')),
            ('TOPPADDING', (0,0), (-1,-1), 5), ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 12), ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(KeepTogether([hdr_tbl, body_tbl]))
        story.append(Spacer(1, 0.16*inch))

    # ── Pages 3+: Charts ───────────────────────────────────────────────────────
    chart_specs = [
        ('PCA 2-D Projection',
         'Each point represents one customer, coloured by cluster assignment. '
         'Tight, well-separated groups indicate strong cluster structure.',
         expand_fig_height(fig_pca_scatter(feature_matrix, labels, k), 6.5)),
        ('Cluster Size Distribution',
         'Number of customers per cluster. Large imbalances may suggest K is too high.',
         expand_fig_height(fig_cluster_sizes(labels, k), 6.5)),
        ('Feature Heatmap — Top 15 Differentiators',
         'Z-scored cluster means. Green = above average for that feature; '
         'Red = below average. Features are ranked by variance across clusters.',
         pdf_heatmap()),
        ('Silhouette Analysis',
         'Per-customer silhouette coefficients by cluster. The red dashed line is '
         'the overall average. Wider positive bands indicate tighter, more distinct clusters.',
         expand_fig_height(fig_silhouette(feature_matrix, labels, k), 6.5)),
    ]

    for title, caption, fig in chart_specs:
        story.append(PageBreak())
        story.append(Paragraph(title,
                                ps(f'CT_{title}', fontName='Helvetica-Bold',
                                   fontSize=14, textColor=C_NAVY, spaceAfter=4)))
        story.append(HRFlowable(width='100%', thickness=1.5, color=C_BLUE, spaceAfter=6))
        story.append(Paragraph(caption,
                                ps(f'CC_{title}', fontSize=9, textColor=C_GRAY,
                                   leading=13, spaceAfter=12)))
        max_h = PAGE_H - 3.2*inch
        aspect = fig.get_figheight() / fig.get_figwidth()
        img_w = CW
        img_h = img_w * aspect
        if img_h > max_h:
            img_h = max_h
            img_w = img_h / aspect
        story.append(fig_img(fig, img_w))

    doc.build(story)
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
cmap = plt.colormaps[PALETTE].resampled(k)

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
