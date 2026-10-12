"""实践 3 拓展分析（题面之外，对主结果做稳健性检验与应用）
依赖 分析.py 里已经算好的月度超额收益 ex_m_all、市场 mk_m 等变量。
输出：结果/06–11 的 csv 与 fig-04 至 fig-06。
"""
if "ex_m_all" not in globals():
    from 分析 import *            # noqa: F401,F403  脚本方式运行时载入主分析的变量

from itertools import combinations
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson, jarque_bera

rng = np.random.default_rng(2026)
n_m = len(mk_m)
x_all = mk_m.values
Y = {c: ex_m_all[c].values for c in ORDER}

# 月度无风险收益（由总收益与超额收益反推）
mk_total = (1 + mkt).groupby(mkt.index.to_period("M")).prod() - 1
rf_m = mk_total.values - mk_m.values


# ---------- 拓展 1：回归诊断与稳健标准误 ----------
rows = []
B = 5000
for c in ORDER:
    y, x = Y[c], x_all
    X = sm.add_constant(x)
    ols = sm.OLS(y, X).fit()
    hc3 = sm.OLS(y, X).fit(cov_type="HC3")
    nw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    # 配对自助法（pairs bootstrap）
    ii = rng.integers(0, n_m, size=(B, n_m))
    xb, yb = x[ii], y[ii]
    xm, ym = xb.mean(1, keepdims=True), yb.mean(1, keepdims=True)
    bb = ((xb - xm) * (yb - ym)).sum(1) / ((xb - xm) ** 2).sum(1)
    ab = ym[:, 0] - bb * xm[:, 0]
    jb_stat, jb_p, skew, kurt = jarque_bera(ols.resid)
    bp_lm, bp_p, _, _ = het_breuschpagan(ols.resid, X)
    rows.append({
        "代码": c, "简称": NAMES[c],
        "beta": ols.params[1], "se_ols": ols.bse[1], "se_hc3": hc3.bse[1], "se_nw": nw.bse[1],
        "beta_boot_lo": np.percentile(bb, 2.5), "beta_boot_hi": np.percentile(bb, 97.5),
        "alpha": ols.params[0], "t_ols": ols.tvalues[0], "t_hc3": hc3.tvalues[0], "t_nw": nw.tvalues[0],
        "alpha_boot_lo": np.percentile(ab, 2.5), "alpha_boot_hi": np.percentile(ab, 97.5),
        "JB_p": jb_p, "skew": skew, "kurt": kurt, "BP_p": bp_p, "DW": durbin_watson(ols.resid),
    })
diag = pd.DataFrame(rows).set_index("代码")
diag.round(5).to_csv(OUT / "06-诊断与稳健标准误-月度.csv", encoding="utf-8-sig")


# ---------- 拓展 2：β 的稳定性与预测 ----------
half = n_m // 2
rows = []
for c in ORDER:
    y, x = Y[c], x_all
    m1 = sm.OLS(y[:half], sm.add_constant(x[:half])).fit()
    m2 = sm.OLS(y[half:], sm.add_constant(x[half:])).fit()
    D = (np.arange(n_m) >= half).astype(float)
    mi = sm.OLS(y, np.column_stack([np.ones(n_m), x, D, D * x])).fit()
    b1, b2 = m1.params[1], m2.params[1]
    rows.append({
        "代码": c, "简称": NAMES[c], "beta_前半": b1, "se_前半": m1.bse[1],
        "beta_后半": b2, "se_后半": m2.bse[1],
        "差(后-前)": b2 - b1, "t_差": mi.tvalues[3], "p_差": mi.pvalues[3],
        "Blume调整_前半": 2 / 3 * b1 + 1 / 3,
        "误差_未调整": abs(b2 - b1), "误差_调整后": abs(b2 - (2 / 3 * b1 + 1 / 3)),
    })
stab = pd.DataFrame(rows).set_index("代码")
stab.round(4).to_csv(OUT / "07-β稳定性与Blume调整-月度.csv", encoding="utf-8-sig")

W_ROLL = 36
roll = pd.DataFrame({c: pd.Series(Y[c]).rolling(W_ROLL).cov(pd.Series(x_all))
                     / pd.Series(x_all).rolling(W_ROLL).var() for c in ORDER})
roll.index = ex_m_all.index
roll.rename(columns=NAMES).round(4).to_csv(OUT / "07b-滚动β-36月.csv", encoding="utf-8-sig")

# Merrill Lynch：用总收益率（不减无风险利率）回归，斜率仍是 β，截距 = α + (1-β) r_f
rows = []
mk_tot_m = mk_m.values + rf_m
for c in ORDER:
    y_tot = Y[c] + rf_m
    m = sm.OLS(y_tot, sm.add_constant(mk_tot_m)).fit()
    b = tab_m.loc[c, "beta"]; a = tab_m.loc[c, "alpha"]
    rows.append({"代码": c, "简称": NAMES[c], "beta_超额收益": b, "beta_总收益": m.params[1],
                 "a_总收益回归截距": m.params[0], "alpha+(1-beta)*rf均值": a + (1 - b) * rf_m.mean()})
ml = pd.DataFrame(rows).set_index("代码")
ml.round(5).to_csv(OUT / "07c-MerrillLynch总收益回归-月度.csv", encoding="utf-8-sig")


# ---------- 拓展 3：α 与 β 的分离（对冲）----------
def perf(r):
    r = np.asarray(r)
    mu, sd = r.mean(), r.std(ddof=1)
    return {"月均超额收益": mu, "年化超额收益": mu * 12, "年化波动": sd * np.sqrt(12),
            "年化夏普": mu / sd * np.sqrt(12), "t(均值)": mu / (sd / np.sqrt(len(r)))}

rows = []
for c in ORDER:
    y, b = Y[c], tab_m.loc[c, "beta"]
    h = y - b * x_all
    rows.append({"代码": c, "简称": NAMES[c], "组合": "未对冲", **perf(y),
                 "与市场相关": np.corrcoef(y, x_all)[0, 1]})
    rows.append({"代码": c, "简称": NAMES[c], "组合": "样本内β对冲", **perf(h),
                 "与市场相关": np.corrcoef(h, x_all)[0, 1]})
    # 样本外：只用 t 月之前的数据估 β（扩张窗口，最少 24 个月）
    ts = range(24, n_m)
    bt = np.array([np.cov(y[:t], x_all[:t], ddof=1)[0, 1] / np.var(x_all[:t], ddof=1) for t in ts])
    ho = y[24:] - bt * x_all[24:]
    rows.append({"代码": c, "简称": NAMES[c], "组合": "样本外对冲（第25–60月）", **perf(ho),
                 "与市场相关": np.corrcoef(ho, x_all[24:])[0, 1]})
    rows.append({"代码": c, "简称": NAMES[c], "组合": "未对冲（第25–60月）", **perf(y[24:]),
                 "与市场相关": np.corrcoef(y[24:], x_all[24:])[0, 1]})
hedge = pd.DataFrame(rows)
hedge.round(4).to_csv(OUT / "08-α转移对冲-月度.csv", index=False, encoding="utf-8-sig")


# ---------- 拓展 4：等权组合与分散化 ----------
rows = []
curve = {}
for N in range(1, 6):
    vals = []
    for sub in combinations(ORDER, N):
        p = ex_m_all[list(sub)].mean(axis=1).values
        m = sm.OLS(p, sm.add_constant(x_all)).fit()
        tv = float(np.var(p, ddof=1)); sv = m.params[1] ** 2 * float(np.var(x_all, ddof=1))
        vals.append((tv, sv, (m.resid ** 2).sum() / (n_m - 1)))
    a_ = np.array(vals).mean(0)
    curve[N] = a_
    rows.append({"N": N, "组合数": len(vals), "平均总方差": a_[0], "平均系统方差": a_[1],
                 "平均非系统方差": a_[2], "平均总标准差(%)": np.sqrt(a_[0]) * 100})
div = pd.DataFrame(rows).set_index("N")
div["非系统方差_理论(单只平均/N)"] = div.loc[1, "平均非系统方差"] / div.index
div.round(6).to_csv(OUT / "09-等权组合分散化-月度.csv", encoding="utf-8-sig")

p5 = ex_m_all[ORDER].mean(axis=1).values
m5 = sm.OLS(p5, sm.add_constant(x_all)).fit()
port5 = {"alpha": m5.params[0], "t_alpha": m5.tvalues[0], "beta": m5.params[1], "se_beta": m5.bse[1],
         "R2": m5.rsquared, "beta_mean_of_stocks": tab_m["beta"].mean(), "alpha_mean_of_stocks": tab_m["alpha"].mean(),
         "nonsys_var": (m5.resid ** 2).sum() / (n_m - 1),
         "nonsys_var_if_uncorrelated": (tab_m["nonsys_var"].sum()) / 25}
pd.Series(port5).round(6).to_csv(OUT / "09b-五只等权组合回归-月度.csv", header=["值"], encoding="utf-8-sig")


# ---------- 拓展 5：市场代理的敏感性（Roll 批评）----------
PROXIES = {"000300": "沪深300", "000001": "上证综指", "399106": "深证综指",
           "000002": "上证A股", "399107": "深证A股", "000903": "中证100"}
rows = []
key = trade_days.to_period("M")
for code, nm in PROXIES.items():
    r = idx[idx.Indexcd == code].set_index("Trddt")["Retindex"].loc[START:END].reindex(trade_days)
    pm = (1 + r).groupby(key).prod().values - 1
    xm = pm - rf_m
    for c in ORDER:
        m = sm.OLS(Y[c], sm.add_constant(xm)).fit()
        rows.append({"市场代理": nm, "代码": c, "简称": NAMES[c], "beta": m.params[1], "se_beta": m.bse[1],
                     "alpha": m.params[0], "t_alpha": m.tvalues[0], "R2": m.rsquared,
                     "代理月均超额收益": xm.mean(), "代理月波动": xm.std(ddof=1)})
proxy = pd.DataFrame(rows)
proxy.round(4).to_csv(OUT / "10-市场代理敏感性-月度.csv", index=False, encoding="utf-8-sig")

# 不同市场代理下的 SML（5 点截面回归）
avg_ex = pd.Series({c: Y[c].mean() for c in ORDER})
rows = []
for nm in PROXIES.values():
    sub = proxy[proxy["市场代理"] == nm].set_index("代码").loc[ORDER]
    m = sm.OLS(avg_ex.loc[ORDER].values, sm.add_constant(sub["beta"].values)).fit()
    rows.append({"市场代理": nm, "代理月均超额收益": sub["代理月均超额收益"].iloc[0],
                 "gamma0": m.params[0], "t_gamma0": m.tvalues[0],
                 "gamma1": m.params[1], "t_gamma1": m.tvalues[1], "R2": m.rsquared})
sml_proxy = pd.DataFrame(rows)
sml_proxy.round(4).to_csv(OUT / "10b-SML-不同市场代理-月度.csv", index=False, encoding="utf-8-sig")


# ---------- 拓展 6：绩效指标（夏普、特雷诺、詹森 α、信息比率）----------
rows = []
for c in ORDER:
    y = Y[c]; b = tab_m.loc[c, "beta"]; a = tab_m.loc[c, "alpha"]; se = tab_m.loc[c, "sigma_e"]
    rows.append({"代码": c, "简称": NAMES[c], "月均超额收益": y.mean(), "月标准差": y.std(ddof=1),
                 "年化夏普": y.mean() / y.std(ddof=1) * np.sqrt(12),
                 "特雷诺(月)": y.mean() / b if abs(b) > 0.05 else np.nan,
                 "詹森alpha(月)": a, "信息比率(年化)": a / se * np.sqrt(12)})
rows.append({"代码": INDEX_CODE, "简称": "沪深300", "月均超额收益": x_all.mean(), "月标准差": x_all.std(ddof=1),
             "年化夏普": x_all.mean() / x_all.std(ddof=1) * np.sqrt(12), "特雷诺(月)": x_all.mean(),
             "詹森alpha(月)": 0.0, "信息比率(年化)": np.nan})
perfm = pd.DataFrame(rows).set_index("代码")
perfm.round(5).to_csv(OUT / "11-绩效指标-月度.csv", encoding="utf-8-sig")


# ---------- 图 ----------
fig, ax = plt.subplots(figsize=(8, 4.3))
for c in ORDER:
    ax.plot(roll.index, roll[c], label=NAMES[c])
ax.axhline(1, color="gray", lw=.6, ls="--"); ax.axhline(0, color="gray", lw=.5)
ax.set_xlabel("窗口结束月份"); ax.set_ylabel("滚动 β（36 个月）")
ax.set_title("滚动 β：窗口 36 个月，月度超额收益"); ax.legend(ncol=3, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig-04-滚动β-36月.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 4.3))
ax.plot(div.index, div["平均系统方差"] * 1e4, "o-", label="系统方差（平均）")
ax.plot(div.index, div["平均非系统方差"] * 1e4, "s-", label="非系统方差（实际，平均）")
ax.plot(div.index, div["非系统方差_理论(单只平均/N)"] * 1e4, "^--", color="gray", label="非系统方差（残差互不相关时的理论值 σ²(e)/N）")
ax.set_xticks(div.index); ax.set_xlabel("组合中股票只数 N（对所有 N 只组合取平均）"); ax.set_ylabel("月度方差（%²）")
ax.legend(fontsize=8); ax.set_title("等权组合的分散化")
fig.tight_layout(); fig.savefig(OUT / "fig-05-等权组合分散化.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.3))
c = "601899"
y = Y[c]; b = tab_m.loc[c, "beta"]
bt = np.array([np.cov(y[:t], x_all[:t], ddof=1)[0, 1] / np.var(x_all[:t], ddof=1) for t in range(24, n_m)])
ho = y[24:] - bt * x_all[24:]
dates = ex_m_all.index[24:]
ax.plot(dates, np.cumsum(y[24:]) * 100, label="紫金矿业 未对冲（累计超额收益）")
ax.plot(dates, np.cumsum(ho) * 100, label="紫金矿业 对冲（滚动 β，样本外）")
ax.plot(dates, np.cumsum(x_all[24:]) * 100, color="gray", ls="--", label="沪深300 超额收益")
ax.axhline(0, color="gray", lw=.5)
ax.set_ylabel("累计月超额收益（%，简单加总）"); ax.legend(fontsize=9)
ax.set_title("α 转移：做多紫金矿业、做空 β 倍沪深300（第 25–60 个月）")
fig.tight_layout(); fig.savefig(OUT / "fig-06-α转移对冲-紫金矿业.png", dpi=150); plt.close(fig)

# ---------- 摘要 ----------
pd.set_option("display.width", 220, "display.max_columns", 40)
print("== 诊断与稳健标准误"); print(diag[["简称", "beta", "se_ols", "se_hc3", "se_nw", "beta_boot_lo", "beta_boot_hi", "t_ols", "t_hc3", "t_nw", "alpha_boot_lo", "alpha_boot_hi", "JB_p", "BP_p", "DW"]].round(3))
print("== 稳定性"); print(stab.round(3))
print("== MerrillLynch"); print(ml.round(4))
print("== 对冲"); print(hedge.round(3).to_string(index=False))
print("== 分散化"); print(div.round(5)); print(pd.Series(port5).round(5))
print("== 代理"); print(proxy.pivot(index="简称", columns="市场代理", values="beta").round(2)); print(proxy.pivot(index="简称", columns="市场代理", values="R2").round(2))
print("== SML按代理"); print(sml_proxy.round(4))
print("== 绩效"); print(perfm.round(4))
