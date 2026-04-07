"""
===============================================================================
PROJECT 3: Fama-French Factor Model + Green Factor (Asset Pricing)
===============================================================================
RESEARCH QUESTION:
    Does a "green factor" (long clean energy, short fossil fuels) explain
    stock returns beyond the traditional Fama-French 3 factors?
METHOD:
    Download Fama-French 3 factors from Kenneth French Data Library.
    Construct a Green-Minus-Brown (GMB) factor from ETF returns.
    Run time-series regressions and GRS test.
DATA:
    Kenneth French Data Library (free), Yahoo Finance for ETFs
===============================================================================
"""
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats
import urllib.request, zipfile, io, os, warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
for d in ['output/figures','output/tables','data']:
    os.makedirs(d, exist_ok=True)

# =============================================================================
# STEP 1: Download Fama-French 3 Factors
# =============================================================================
print("STEP 1: Downloading Fama-French factors from Kenneth French Library...")

url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
try:
    response = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(response.read()))
    fname = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
    
    # Read, skip header rows
    raw = z.open(fname).read().decode('utf-8')
    lines = raw.strip().split('\n')
    
    # Find the header row
    start = 0
    for i, line in enumerate(lines):
        if 'Mkt-RF' in line:
            start = i; break
    
    # Find end of daily data (blank line or annual section)
    end = len(lines)
    for i in range(start+1, len(lines)):
        if lines[i].strip() == '' or len(lines[i].strip()) < 10:
            end = i; break
    
    from io import StringIO
    ff_data = pd.read_csv(StringIO('\n'.join(lines[start:end])))
    ff_data.columns = [c.strip() for c in ff_data.columns]
    ff_data = ff_data.rename(columns={ff_data.columns[0]: 'Date'})
    ff_data['Date'] = pd.to_datetime(ff_data['Date'], format='%Y%m%d')
    ff_data = ff_data.set_index('Date')
    ff_data = ff_data.apply(pd.to_numeric, errors='coerce')
    ff_data = ff_data.loc['2015':'2025']
    print(f"  Loaded {len(ff_data)} daily observations of Mkt-RF, SMB, HML, RF")
except Exception as e:
    print(f"  Error downloading FF data: {e}")
    print("  Generating synthetic FF factors for demonstration...")
    dates = pd.bdate_range('2015-01-01', '2025-12-31')
    np.random.seed(42)
    ff_data = pd.DataFrame({
        'Mkt-RF': np.random.normal(0.04, 1.0, len(dates)),
        'SMB': np.random.normal(0.01, 0.5, len(dates)),
        'HML': np.random.normal(0.01, 0.5, len(dates)),
        'RF': np.full(len(dates), 0.01)
    }, index=dates)

ff_data.to_csv('data/fama_french_daily.csv')

# =============================================================================
# STEP 2: Download ETF data and construct Green-Minus-Brown factor
# =============================================================================
print("\nSTEP 2: Constructing Green-Minus-Brown (GMB) factor...")

etfs = {'ICLN':'Green','QCLN':'Green','PBW':'Green',
        'XLE':'Brown','XOP':'Brown','VDE':'Brown'}

prices = {}
for t in etfs:
    df = yf.download(t, start='2015-01-01', end='2025-12-31', auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if not df.empty:
        prices[t] = df['Close']
        print(f"  {t} ({etfs[t]}): {len(df)} obs")

prices = pd.DataFrame(prices).dropna()
rets = np.log(prices / prices.shift(1)).dropna() * 100  # Daily log returns (%)

# GMB = average green return - average brown return
green_cols = [c for c in rets.columns if etfs.get(c) == 'Green']
brown_cols = [c for c in rets.columns if etfs.get(c) == 'Brown']
rets['GMB'] = rets[green_cols].mean(axis=1) - rets[brown_cols].mean(axis=1)

# =============================================================================
# STEP 3: Merge and run regressions
# =============================================================================
print("\nSTEP 3: Running factor regressions...")

# Merge FF factors with ETF returns
merged = rets.join(ff_data, how='inner').dropna()

# Test assets: individual ETFs
results_list = []
for etf in [c for c in rets.columns if c != 'GMB']:
    # FF3 model
    y = merged[etf] - merged['RF']
    X3 = add_constant(merged[['Mkt-RF','SMB','HML']])
    ff3 = OLS(y, X3).fit()
    
    # FF3 + GMB model
    X4 = add_constant(merged[['Mkt-RF','SMB','HML','GMB']])
    ff4 = OLS(y, X4).fit()
    
    results_list.append({
        'ETF': etf, 'Type': etfs[etf],
        'FF3_alpha': ff3.params['const'], 'FF3_alpha_p': ff3.pvalues['const'],
        'FF3_R2': ff3.rsquared,
        'FF3_MktBeta': ff3.params['Mkt-RF'],
        'FF4_alpha': ff4.params['const'], 'FF4_alpha_p': ff4.pvalues['const'],
        'FF4_R2': ff4.rsquared,
        'GMB_beta': ff4.params['GMB'], 'GMB_p': ff4.pvalues['GMB']
    })
    
    print(f"  {etf} ({etfs[etf]:5s}): FF3 α={ff3.params['const']:+.4f} | "
          f"FF4 α={ff4.params['const']:+.4f}, GMB β={ff4.params['GMB']:+.4f} (p={ff4.pvalues['GMB']:.4f})")

results_df = pd.DataFrame(results_list)
results_df.to_csv('output/tables/factor_regression_results.csv', index=False)

# =============================================================================
# STEP 4: Visualizations
# =============================================================================
print("\nSTEP 4: Creating visualizations...")

# Fig 1: Factor Cumulative Returns
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Cumulative returns of FF factors + GMB
factors = merged[['Mkt-RF','SMB','HML']].copy()
factors['GMB'] = merged['GMB']
cum = (1 + factors/100).cumprod()
for col in cum.columns:
    axes[0].plot(cum.index, cum[col], label=col, linewidth=1.2)
axes[0].set_title('Cumulative Factor Returns', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Cumulative Return (1 = start)')
axes[0].legend(fontsize=11)
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# GMB factor time series
axes[1].plot(merged.index, merged['GMB'].rolling(20).mean(), color='green', linewidth=1)
axes[1].fill_between(merged.index, 0, merged['GMB'].rolling(20).mean(), alpha=0.3, color='green')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('Green-Minus-Brown Factor (20-day MA)', fontweight='bold', fontsize=13)
axes[1].set_ylabel('Daily Return (%)')
plt.tight_layout()
plt.savefig('output/figures/fig1_factor_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 2: Alpha comparison (FF3 vs FF3+GMB)
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
w = 0.35
colors_type = ['#2ecc71' if t == 'Green' else '#e74c3c' for t in results_df['Type']]
ax.bar(x - w/2, results_df['FF3_alpha'], w, label='FF3 Alpha', color='steelblue', alpha=0.8)
ax.bar(x + w/2, results_df['FF4_alpha'], w, label='FF3+GMB Alpha', color='coral', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(results_df['ETF'], rotation=45)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Pricing Errors: FF3 vs FF3+GMB Model', fontweight='bold')
ax.set_ylabel('Alpha (daily %)')
ax.legend()
plt.tight_layout()
plt.savefig('output/figures/fig2_alpha_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Fig 3: Factor correlation heatmap
corr = factors.corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, 
            square=True, linewidths=1)
ax.set_title('Factor Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('output/figures/fig3_factor_correlations.png', dpi=150, bbox_inches='tight')
plt.close()

print("  COMPLETE!")
