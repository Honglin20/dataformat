#!/usr/bin/env python3
"""Generate QSNR summary HTML table from experiment results."""

import csv
from pathlib import Path

def read_csv(filename):
    data = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fmt = row['format']
            dist = row['distribution']
            sqnr = float(row['SQNR_dB'])
            if fmt not in data:
                data[fmt] = {}
            data[fmt][dist] = sqnr
    return data

data_4bit = read_csv('results/exp1/results_4bit.csv')
data_8bit = read_csv('results/exp1/results_8bit.csv')

# Define distributions grouped by family
dist_groups = [
    ("Gaussian",       ["Gauss(σ=0.5)", "Gauss(σ=1.0)", "Gauss(σ=2.0)", "Gauss(σ=5.0)"]),
    ("Laplace",        ["Laplace(b=0.5)", "Laplace(b=1.0)", "Laplace(b=2.0)"]),
    ("Student-t",      ["Student-t(ν=3)", "Student-t(ν=5)", "Student-t(ν=10)"]),
    ("Bimodal",        ["Bimodal(μ=±2.0)", "Bimodal(μ=±3.0)", "Bimodal(μ=±5.0)"]),
    ("Chan. Outlier",  ["ChanOut(σ=30)", "ChanOut(σ=50)", "ChanOut(σ=100)"]),
    ("Spiky Outlier",  ["Spiky(10×)", "Spiky(50×)", "Spiky(100×)"]),
    ("Log-Normal",     ["LogNorm(σ=1.0)", "LogNorm(σ=2.0)"]),
    ("Uniform",        ["Uniform(±1.0)", "Uniform(±3.0)"]),
]
all_dists = [d for _, dists in dist_groups for d in dists]

# Format display names and order
formats_4bit = [
    ("FP16",              "FP16"),
    ("SQ-Format-INT4",    "SQ-FORMAT-INT"),
    ("SQ-Format-FP4",     "SQ-FORMAT-FP"),
    ("INT4-PerTensor",    "INT4(TENSOR)"),
    ("INT4-PerChannel",   "INT4(CHANNEL)"),
    ("HAD+INT4-PerTensor","HAD+INT4(T)"),
    ("HAD+INT4-PerChannel","HAD+INT4(C)"),
    ("MXINT4",            "MXINT4"),
    ("MXFP4",             "MXFP4"),
]

formats_8bit = [
    ("FP16",              "FP16"),
    ("SQ-Format-INT8",    "SQ-FORMAT-INT"),
    ("SQ-Format-FP8",     "SQ-FORMAT-FP"),
    ("INT8-PerTensor",    "INT8(TENSOR)"),
    ("INT8-PerChannel",   "INT8(CHANNEL)"),
    ("HAD+INT8-PerTensor","HAD+INT8(T)"),
    ("HAD+INT8-PerChannel","HAD+INT8(C)"),
    ("MXINT8",            "MXINT8"),
    ("MXFP8",             "MXFP8"),
]

def get_color(val, vmin, vmax):
    """Return CSS color from red (low) to green (high)."""
    if vmax == vmin:
        ratio = 0.5
    else:
        ratio = (val - vmin) / (vmax - vmin)
    ratio = max(0.0, min(1.0, ratio))
    # red -> yellow -> green
    if ratio < 0.5:
        r = 255
        g = int(ratio * 2 * 220)
        b = int(60 * (1 - ratio * 2))
    else:
        r = int((1 - (ratio - 0.5) * 2) * 220)
        g = 200
        b = int(60 * (ratio - 0.5) * 2)
    return f"rgb({r},{g},{b})"

def build_table(data, formats, bits_label):
    # Collect all values for global color scale (excluding FP16 as reference)
    non_ref_vals = []
    for fmt_key, _ in formats:
        if fmt_key == "FP16":
            continue
        for d in all_dists:
            v = data.get(fmt_key, {}).get(d)
            if v is not None:
                non_ref_vals.append(v)
    vmin, vmax = min(non_ref_vals), max(non_ref_vals)

    rows = []
    rows.append(f'<h2>{bits_label} — SQNR (dB) by Format and Distribution</h2>')
    rows.append('<div class="table-wrap"><table>')

    # Header: group row
    rows.append('<thead>')
    rows.append('<tr><th rowspan="2" class="fmt-col">Format</th>')
    for grp_name, dists in dist_groups:
        rows.append(f'<th colspan="{len(dists)}" class="grp-hdr">{grp_name}</th>')
    rows.append('<th rowspan="2" class="avg-col">Avg</th></tr>')

    # Header: individual distributions
    rows.append('<tr>')
    for _, dists in dist_groups:
        for d in dists:
            short = d.replace("Gauss(σ=","σ=").replace(")","").replace("Laplace(b=","b=") \
                     .replace("Student-t(ν=","ν=").replace("Bimodal(μ=","μ=") \
                     .replace("ChanOut(σ=","σ=").replace("Spiky(","").replace("×","×") \
                     .replace("LogNorm(σ=","σ=").replace("Uniform(±","±")
            rows.append(f'<th class="dist-hdr" title="{d}">{short}</th>')
    rows.append('</tr>')
    rows.append('</thead><tbody>')

    for fmt_key, fmt_label in formats:
        is_ref = fmt_key == "FP16"
        row_class = ' class="ref-row"' if is_ref else ''
        rows.append(f'<tr{row_class}>')
        rows.append(f'<td class="fmt-name">{fmt_label}</td>')
        vals = []
        for d in all_dists:
            v = data.get(fmt_key, {}).get(d)
            if v is not None:
                vals.append(v)
                if is_ref:
                    rows.append(f'<td class="ref-cell">{v:.1f}</td>')
                else:
                    color = get_color(v, vmin, vmax)
                    rows.append(f'<td style="background:{color};color:#000">{v:.1f}</td>')
            else:
                rows.append('<td class="na">—</td>')
        if vals:
            avg = sum(vals) / len(vals)
            if is_ref:
                rows.append(f'<td class="ref-cell avg-cell">{avg:.1f}</td>')
            else:
                color = get_color(avg, vmin, vmax)
                rows.append(f'<td style="background:{color};color:#000;font-weight:bold">{avg:.1f}</td>')
        rows.append('</tr>')

    rows.append('</tbody></table></div>')
    return '\n'.join(rows)

table_4bit = build_table(data_4bit, formats_4bit, "4-bit Formats")
table_8bit = build_table(data_8bit, formats_8bit, "8-bit Formats")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QSNR Summary — Quantization Format Comparison</title>
<style>
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #f5f7fa;
    color: #222;
    margin: 0;
    padding: 20px;
  }}
  h1 {{
    text-align: center;
    font-size: 1.6em;
    margin-bottom: 4px;
    color: #1a2a4a;
  }}
  .subtitle {{
    text-align: center;
    color: #666;
    margin-bottom: 30px;
    font-size: 0.92em;
  }}
  h2 {{
    font-size: 1.15em;
    color: #1a2a4a;
    margin: 32px 0 10px 0;
    border-left: 4px solid #3a7bd5;
    padding-left: 10px;
  }}
  .table-wrap {{
    overflow-x: auto;
    border-radius: 8px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    background: #fff;
    margin-bottom: 40px;
  }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 0.78em;
  }}
  th, td {{
    padding: 5px 7px;
    text-align: center;
    border: 1px solid #dde2eb;
    white-space: nowrap;
  }}
  th.fmt-col {{
    background: #1a2a4a;
    color: #fff;
    font-size: 0.85em;
    min-width: 110px;
    text-align: left;
    padding-left: 10px;
  }}
  th.grp-hdr {{
    background: #2c4a7e;
    color: #fff;
    font-size: 0.82em;
    letter-spacing: 0.03em;
  }}
  th.dist-hdr {{
    background: #3a6ab5;
    color: #fff;
    font-size: 0.72em;
    font-weight: normal;
    max-width: 55px;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  th.avg-col {{
    background: #1a2a4a;
    color: #ffd700;
    font-size: 0.85em;
    min-width: 48px;
  }}
  td.fmt-name {{
    font-weight: 600;
    background: #f0f4fa;
    text-align: left;
    padding-left: 10px;
    color: #1a2a4a;
    border-right: 2px solid #ccd;
  }}
  td.ref-cell {{
    background: #e8eefc;
    color: #444;
    font-style: italic;
  }}
  tr.ref-row td.fmt-name {{
    background: #dce6f7;
    color: #1a2a4a;
  }}
  td.na {{ color: #bbb; background: #fafafa; }}
  td.avg-cell {{ border-left: 2px solid #aab; }}
  .legend {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 10px 0 20px 0;
    font-size: 0.82em;
    color: #555;
  }}
  .legend-bar {{
    width: 160px;
    height: 14px;
    background: linear-gradient(to right, rgb(255,60,60), rgb(255,220,0), rgb(0,200,100));
    border-radius: 3px;
    border: 1px solid #ccc;
  }}
  .note {{
    font-size: 0.8em;
    color: #888;
    margin-top: -10px;
    margin-bottom: 16px;
  }}
</style>
</head>
<body>
<h1>QSNR (dB) Summary — Quantization Format Comparison</h1>
<p class="subtitle">Signal-to-Quantization-Noise Ratio across distributions. Higher is better. FP16 is reference baseline.</p>

<div class="legend">
  <span>Low SQNR</span>
  <div class="legend-bar"></div>
  <span>High SQNR</span>
  <span style="margin-left:16px;color:#999">| Color scale: per-section (non-FP16 only)</span>
</div>
<p class="note">Column headers are abbreviated; hover for full name. "Avg" = mean SQNR across all 24 test distributions.</p>

{table_4bit}
{table_8bit}

<p style="font-size:0.75em;color:#aaa;margin-top:30px;text-align:right">
  Generated from results/exp1/results_4bit.csv &amp; results_8bit.csv
</p>
</body>
</html>
"""

out_path = Path('results/qsnr_summary.html')
out_path.write_text(html)
print(f"Written to {out_path}")
print(f"4-bit formats: {[f for f,_ in formats_4bit]}")
print(f"8-bit formats: {[f for f,_ in formats_8bit]}")
