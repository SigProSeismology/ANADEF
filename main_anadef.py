#!/usr/bin/env python3
"""
ANADEF pipeline v2.1 - dual-parameter (b-value + ETAS background rate) alarm forecasting.

    python main_anadef.py --config Configuration_Parameters_Hokkaido.json

Stages
  0  load catalog (minute-resolution times, one magnitude convention, duplicates dropped)
  1  grid (optionally fixed box) with cos-latitude cell weights
  2  per window: Mc -> truncate -> ETAS MLE (bounds checked, SEs) -> Zhuang declustering with
     spatially varying mu -> mu(x,y) [events/deg^2/yr] and b(x,y) [Aki-Utsu, sigma mask]
  3  calibration window: Molchan sweeps, joint 2-D calibration on unstandardised PD, Table 3
     (S-score + bootstrap CI), Fig. 3/4/5, nested permutation test with two nulls, alarm maps
  4  forward window: fields from the latest catalog, thresholds FROZEN from the calibration
     window (absolute b_opt, mu_opt), no re-optimisation, no scaling

v2.1 additions (Hokkaido)
  * b-value map with a cell-wise completeness magnitude (b plateau criterion) instead of one global Mc
  * kernel-smoothed all-event density as a Level-1b benchmark next to the ETAS background rate
  * cluster (cell) bootstrap for the S-score confidence intervals
  * split-half out-of-sample check of the calibrated operating points
"""

import argparse
import json
import math
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd

from config.config_parser import Config
from src.data_loader import load_and_preprocess_catalog, create_spatial_grid
from src.mc_estimation import validate_mc_estimation
from src.b_value import calculate_b_value_map, calculate_b_value_map_cellwise
from src.zhuang_declustering import (build_neighbors, fit_etas, stochastic_declustering,
                                     background_rate_on_grid, branching_ratio, YEAR_DAYS)
from src.forecasting import (assign_events_to_cells, clean_fields, joint_alarm, metrics, evaluate_models,
                             nested_permutation_test, calibrate_joint, calibrate_single, split_half_validation)
from src.shapefile_exporter import save_three_alarm_shapefiles
from src import visualization as viz


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient='records')
    return str(o)


def dump(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


# ----------------------------------------------------------------------------- stage 2
def compute_fields(df_all: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, cfg: Config,
                   centers: np.ndarray, label: str) -> Dict:
    print(f"\n{'=' * 70}\nFIELDS {label}: {t0.date()} .. {t1.date()}\n{'=' * 70}")
    df = df_all[(df_all['time'] >= t0) & (df_all['time'] <= t1)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"no events in {label}")
    # Mc is estimated on the NATIVE (binned) catalog magnitudes and then converted, because MAXC on a
    # linearly rescaled magnitude is aliased by the bin structure.  The Utsu correction uses the
    # converted bin width slope*0.1.
    bin_eff = cfg.bin_width_mc * (cfg.mag_slope if cfg.mag_convert else 1.0)
    if cfg.fixed_mc is None:
        Mc_native = validate_mc_estimation(df['magnitude_catalog'].values, cfg.bin_width_mc)
        Mc = cfg.mag_slope * Mc_native + cfg.mag_intercept if cfg.mag_convert else Mc_native
        if cfg.mag_convert:
            print(f"  Mc (native) = {Mc_native:.2f} -> converted Mc = {Mc:.3f}")
    else:
        Mc = float(cfg.fixed_mc); print(f"  Mc fixed by configuration: {Mc:.2f}")
    n_all = len(df)
    mc_floor = float(cfg.b_mc_floor) if (cfg.b_mc_mode == 'cellwise' and cfg.b_mc_floor is not None) else Mc
    df_floor = df[df['magnitude'] >= mc_floor - 1e-9].reset_index(drop=True)
    df = df[df['magnitude'] >= Mc - 1e-9].reset_index(drop=True)
    print(f"  {len(df)} of {n_all} events at M >= Mc = {Mc:.2f} (ETAS / mu); {len(df_floor)} at M >= {mc_floor:.2f} (b-map floor)")

    times = (df['time'] - t0).dt.total_seconds().values / (YEAR_DAYS * 86400)
    T = (t1 - t0).total_seconds() / (YEAR_DAYS * 86400)
    coords = np.ascontiguousarray(df[['longitude', 'latitude']].values, dtype=float)
    mags = df['magnitude'].values.astype(float)
    area = (coords[:, 0].max() - coords[:, 0].min() + 0.4) * (coords[:, 1].max() - coords[:, 1].min() + 0.4)

    flat, ptr, cnt = build_neighbors(times, coords, cfg.space_cut_deg, cfg.time_cut_years)
    print(f"  ETAS: {len(flat)} candidate parent-offspring pairs within {cfg.space_cut_deg} deg / {cfg.time_cut_days:.0f} d")
    import time as _t; _t0 = _t.time()
    params = fit_etas(times, coords, mags, flat, ptr, cnt, area, T, Mc, cfg.time_cut_years, bounds=tuple(cfg.log_param_bounds),
                      compute_se=cfg.compute_standard_errors, bound_tol=cfg.bound_tolerance)
    se = params.get('standard_errors', {})
    fmt = lambda k, s=1: f"{params[k]:.{s}g}" + (f" ± {se[k]:.{s}g}" if k in se and np.isfinite(se[k]) else "")
    print(f"  ETAS MLE: K={fmt('K_yr',3)} (K0_equiv={params['K0_equiv_infinite']:.3g}, offspring/Mc-parent in window={params['offspring_per_Mc_parent_in_window']:.3g}) "
          f"alpha={fmt('alpha',3)} c={fmt('c_days',3)} d p={fmt('p',3)} "
          f"D={fmt('D_deg2',3)} deg2 q={fmt('q',3)} gamma={fmt('gamma',2)} mu0={fmt('mu0_per_deg2_yr',3)} | {params['message']} "
          f"({params['n_iter']} it, {params['n_fev']} fev, {_t.time()-_t0:.0f} s)")
    _t0 = _t.time()

    phi, mu_ev, bw, n_it = stochastic_declustering(params['x_log'], times, coords, mags, flat, ptr, cnt, Mc, T,
                                                   cfg.np_min_for_mu, cfg.eps_min, cfg.max_iter_mu, cfg.tol_mu)
    print(f"  declustering time {_t.time()-_t0:.0f} s")
    mu_grid = background_rate_on_grid(coords, phi, bw, centers, T)
    # Level 1b: the same variable-bandwidth kernel applied to ALL events >= Mc (phi = 1), i.e. plain smoothed seismicity
    smooth_grid = background_rate_on_grid(coords, np.ones_like(phi), bw, centers, T) if cfg.smoothed_seismicity else np.full(len(centers), np.nan)
    if cfg.b_mc_mode == 'cellwise':
        b_vals, b_sig, n_used, mc_cell = calculate_b_value_map_cellwise(
            df_floor, centers, mc_floor, cfg.b_radius_deg, cfg.nearest_N, cfg.min_events_for_b, cfg.b_value_min, cfg.b_value_max,
            cfg.fallback_max_radius_deg, cfg.sigma_b_mask, bin_eff, cfg.b_mc_ceiling, cfg.b_mc_plateau_bins, cfg.b_mc_plateau_tol, cfg.b_mc_margin)
    else:
        b_vals, b_sig, n_used = calculate_b_value_map(df, centers, Mc, cfg.b_radius_deg, cfg.nearest_N, cfg.min_events_for_b,
                                                      cfg.b_value_min, cfg.b_value_max, cfg.fallback_max_radius_deg,
                                                      cfg.sigma_b_mask, bin_eff)
        mc_cell = np.where(np.isfinite(b_vals), Mc, np.nan)
    b_glob = np.log10(np.e) / (mags.mean() - (Mc - bin_eff / 2))
    params.update({
        'global_b': float(b_glob),
        'branching_ratio_theory': branching_ratio(params['K0_equiv_infinite'], params['alpha'], b_glob),
        'branching_ratio_empirical': float(1 - phi.mean()),
        'background_fraction_empirical': float(phi.mean()),
        'declustering_iterations': int(n_it),
        'phi_mean_M5plus': float(phi[mags >= 5.0].mean()) if (mags >= 5.0).any() else None,
        'n_M5plus_in_window': int((mags >= 5.0).sum()),
        'window': [t0.isoformat(), t1.isoformat()],
    })
    print(f"  branching ratio: theory {params['branching_ratio_theory']:.2f}, empirical {params['branching_ratio_empirical']:.2f}; "
          f"mean phi of M>=5 events {params['phi_mean_M5plus']}")
    grid_df = pd.DataFrame({'lon_center': centers[:, 0], 'lat_center': centers[:, 1], 'b': b_vals, 'b_sigma': b_sig,
                            'n_b': n_used, 'Mc_cell': mc_cell, 'mu': mu_grid, 'smooth': smooth_grid})
    params['b_mc_mode'] = cfg.b_mc_mode; params['b_mc_floor'] = mc_floor
    events = df.assign(phi=phi, mu_at_event=mu_ev)
    return dict(grid=grid_df, params=params, Mc=Mc, events=events, label=label, t0=t0, t1=t1)


# ----------------------------------------------------------------------------- stage 3
def evaluate_window(F: Dict, df_test: pd.DataFrame, cfg: Config, lons, lats, w, out_dir: str) -> Dict:
    grid = F['grid']; lab = F['label']
    tlab = f"{df_test['time'].min().year}-{df_test['time'].max().year}"
    print(f"\n{'-' * 70}\nEVALUATION train {lab} -> test {tlab} ({len(df_test)} events)\n{'-' * 70}")
    q = np.r_[np.linspace(0.01, 0.99, cfg.quantile_steps), 1.0]   # 1.0 = criterion switched off
    nx, ny = len(lons), len(lats)
    evals, tables = {}, []
    for Mthr in cfg.forecast_mag_thresholds:
        ev = df_test[(df_test['magnitude'] >= Mthr) & (df_test['magnitude'] < cfg.forecast_mag_max)]
        ids = assign_events_to_cells(ev, lons, lats)
        if len(ids) < 3:
            print(f"  M>={Mthr}: only {len(ids)} targets - skipped"); continue
        E = evaluate_models(grid, ids, w, q, cfg.bootstrap_iterations, cfg.random_seed, 'PD', cfg.bootstrap_mode)
        E['events'] = ev
        if cfg.split_half and len(ids) >= 10:
            E['split_half'] = split_half_validation(grid, ev, ids, w, q)
            for r in E['split_half']:
                print(f"  split-half {r['direction']}: mu  in-sample PD {r['mu_in_PD']:.3f} (tau {r['mu_in_tau']:.2f}) -> out-of-sample "
                      f"tau {r['mu_out_tau']:.2f} nu {r['mu_out_nu']:.2f} PD {r['mu_out_PD']:.3f} | joint in {r['joint_in_PD']:.3f} -> out PD {r['joint_out_PD']:.3f} (q_b {r['joint_q_b']:.2f})")
        evals[Mthr] = E
        t = E['table'].copy(); t.insert(0, 'target', f"M>={Mthr} (N={len(ids)})"); tables.append(t)
        print(f"\n  Table 3, M >= {Mthr}, N = {len(ids)} (distinct cells {len(np.unique(ids))}; bootstrap = {cfg.bootstrap_mode}):")
        print(E['table'][['model', 'S', 'S_ci_low', 'S_ci_high', 'max_PD', 'tau_at_max', 'nu_at_max', 'q_b', 'q_mu', 'b_thr', 'mu_thr']]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table3 = pd.concat(tables, ignore_index=True)
    table3.to_csv(os.path.join(out_dir, f'table3_{lab}_{tlab}.csv'), index=False)

    # paired bootstrap of the skill differences (same resampled cells for every model)
    pb = []
    for Mthr, E in sorted(evals.items()):
        if 'paired_bootstrap' not in E:
            continue
        t = E['paired_bootstrap'].copy(); t.insert(0, 'target', f"M>={Mthr} (N={E['N']})")
        pb.append(t)
    if pb:
        paired = pd.concat(pb, ignore_index=True)
        paired.to_csv(os.path.join(out_dir, f'paired_bootstrap_{lab}_{tlab}.csv'), index=False)
        print(f"\n  Paired bootstrap of Delta S (same cell resamples for every model, {cfg.bootstrap_iterations} replicates):")
        print(paired.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sh = [dict(target=Mthr, **r) for Mthr, E in evals.items() for r in E.get('split_half', [])]
    if sh:
        pd.DataFrame(sh).to_csv(os.path.join(out_dir, f'split_half_{lab}_{tlab}.csv'), index=False)

    # Fig. 4: max PD / PG vs magnitude threshold
    rows = []
    b, mu = clean_fields(grid)
    for Mthr in cfg.fig4_mags:
        ev = df_test[(df_test['magnitude'] >= Mthr) & (df_test['magnitude'] < cfg.forecast_mag_max)]
        ids = assign_events_to_cells(ev, lons, lats)
        if len(ids) < 2:
            continue
        cb = calibrate_single(b, True, ids, w, q); cm = calibrate_single(mu, False, ids, w, q); cj = calibrate_joint(b, mu, ids, w, q)
        rows.append(dict(magnitude=Mthr, N=len(ids), max_PD_b=cb['PD'], max_PD_mu=cm['PD'], max_PD_joint=cj['PD'],
                         max_PG_b=cb['sweep']['PG'].max(), max_PG_mu=cm['sweep']['PG'].max()))
    df4 = pd.DataFrame(rows); df4.to_csv(os.path.join(out_dir, f'figure4_{lab}_{tlab}.csv'), index=False)

    viz.plot_fig3(evals, os.path.join(out_dir, f'Fig3_molchan_{lab}_{tlab}.pdf'))
    viz.plot_fig4(df4, os.path.join(out_dir, f'Fig4_maxPD_PG_{lab}_{tlab}.pdf'))
    viz.plot_fig5(evals, os.path.join(out_dir, f'Fig5_surfaces_{lab}_{tlab}.pdf'))

    # permutation tests
    perms = {}
    for Mthr, E in evals.items():
        ids = assign_events_to_cells(E['events'], lons, lats)
        P = nested_permutation_test(grid, ids, w, q, nx, ny, cfg.permutation_iterations, cfg.random_seed, 'PD')
        Pz = nested_permutation_test(grid, ids, w, q, nx, ny, cfg.permutation_iterations, cfg.random_seed, 'ZPD')
        perms[Mthr] = {'PD': P, 'ZPD': Pz}
        print(f"  permutation M>={Mthr}: PD  obs={P['observed_stat']:.3f} (mu-only {P['mu_only']['PD']:.3f})  "
              f"p_shuffle={P['p_cell_shuffle']:.3f}  p_shift={P['p_toroidal_shift']:.3f}")
        print(f"                       ZPD obs={Pz['observed_stat']:.3f} (mu-only {Pz['mu_only']['ZPD']:.3f})  "
              f"p_shuffle={Pz['p_cell_shuffle']:.3f}  p_shift={Pz['p_toroidal_shift']:.3f}")
        viz.plot_permutation(P, Mthr, os.path.join(out_dir, f'FigP_permutation_PD_M{Mthr}_{lab}_{tlab}.pdf'))
        viz.plot_permutation(Pz, Mthr, os.path.join(out_dir, f'FigP_permutation_ZPD_M{Mthr}_{lab}_{tlab}.pdf'))
    dump({str(k): {s: {kk: vv for kk, vv in v[s].items() if not kk.startswith('null_')} for s in v} for k, v in perms.items()},
         os.path.join(out_dir, f'permutation_{lab}_{tlab}.json'))

    # calibrated joint thresholds at the calibration magnitude -> alarm maps / shapefiles
    Mcal = cfg.calibration_magnitude
    cal = evals[Mcal]['calibration']['joint']
    thresholds = dict(calibration_magnitude=Mcal, b_thr=cal['b_thr'], mu_thr=cal['mu_thr'], q_b=cal['q_b'], q_mu=cal['q_mu'],
                      tau=cal['tau'], nu=cal['nu'], PD=cal['PD'], criterion='unstandardised PD, joint 2-D search',
                      train_window=lab, test_window=tlab, mu_units='events per deg^2 per year')
    dump(thresholds, os.path.join(out_dir, f'thresholds_{lab}_{tlab}.json'))
    g = grid.copy()
    bb, mm = clean_fields(g)
    g['alarm_b'] = (bb <= cal['b_thr']).astype(int); g['alarm_mu'] = (mm >= cal['mu_thr']).astype(int)
    g['alarm_combined'] = (g['alarm_b'] & g['alarm_mu']).astype(int)
    g.to_csv(os.path.join(out_dir, f'alarm_regions_{lab}_{tlab}.csv'), index=False)
    try:
        save_three_alarm_shapefiles(g, grid_res=cfg.grid_res, out_basename=os.path.join(out_dir, f'alarm_regions_{lab}_{tlab}'),
                                    b_threshold=cal['b_thr'], mu_threshold=cal['mu_thr'])
    except Exception as e:
        print(f"  shapefile export skipped: {e}")
    return dict(evals=evals, table3=table3, thresholds=thresholds, perms=perms, alarm_grid=g, tlab=tlab)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="ANADEF v2 pipeline")
    ap.add_argument('--config', default='Configuration_Parameters_Hokkaido.json')
    args = ap.parse_args()
    cfg = Config(args.config)
    out = cfg.output_dir
    print(f"{cfg.project_name} v{cfg.version}\ncatalog: {cfg.csv_path}\noutput : {out}")

    df_all = load_and_preprocess_catalog(cfg.csv_path, cfg.column_mapping, cfg.drop_duplicates, cfg.mag_convert,
                                         cfg.mag_slope, cfg.mag_intercept, cfg.mag_round_to_bin, cfg.bin_width_mc)
    print(f"{len(df_all)} events, {df_all['time'].min()} .. {df_all['time'].max()}, M {df_all['magnitude'].min():.1f}-{df_all['magnitude'].max():.1f}")
    lons, lats, XX, YY, centers, w, bbox = create_spatial_grid(df_all, cfg.grid_res, cfg.grid_bbox, cfg.cos_lat_weighting)
    print(f"grid {cfg.grid_res} deg: {len(lons)} x {len(lats)} = {len(centers)} cells, box {['%.2f' % v for v in bbox]}, "
          f"cos-lat weights {'on' if cfg.cos_lat_weighting else 'off'}")

    fields: Dict[str, Dict] = {}
    summary = dict(config=os.path.basename(cfg.config_path), n_events=len(df_all), grid_cells=len(centers), bbox=bbox,
                   magnitude_conversion=cfg.mag_convert, windows={})
    calib: Optional[Dict] = None

    def get_fields(t0, t1):
        key = f"{t0.date()}_{t1.date()}"
        if key not in fields:
            F = compute_fields(df_all, t0, t1, cfg, centers, Config.label(t0, t1))
            F['grid'].to_csv(os.path.join(out, f"b_mu_grid_{F['label']}.csv"), index=False)
            F['events'].to_csv(os.path.join(out, f"events_phi_{F['label']}.csv"), index=False)
            dump(F['params'], os.path.join(out, f"etas_params_{F['label']}.json"))
            fields[key] = F
        return fields[key]

    for wdw in cfg.windows:
        F = get_fields(wdw['train_start'], wdw['train_end'])
        summary['windows'][F['label']] = {k: F['params'].get(k) for k in
                                          ('Mc', 'n_events', 'b_mc_mode', 'b_mc_floor', 'K_yr', 'K0_equiv_infinite', 'offspring_per_Mc_parent_in_window', 'alpha', 'c_days', 'p', 'D_deg2', 'q', 'gamma', 'params_at_bound',
                                           'global_b', 'branching_ratio_theory', 'branching_ratio_empirical')}
        if wdw['forecast_start'] is not None:
            df_test = df_all[(df_all['time'] >= wdw['forecast_start']) & (df_all['time'] <= wdw['forecast_end'])].reset_index(drop=True)
            assert wdw['forecast_start'] > wdw['train_end'], "train/test windows overlap"
            R = evaluate_window(F, df_test, cfg, lons, lats, w, out)
            summary['windows'][F['label']]['evaluation'] = dict(test_window=R['tlab'], thresholds=R['thresholds'],
                                                              split_half={str(k): E.get('split_half') for k, E in R['evals'].items()},
                                                              table3=R['table3'].drop(columns=[]).to_dict(orient='records'),
                                                              paired_bootstrap={str(k): (E['paired_bootstrap'].to_dict(orient='records')
                                                                                         if 'paired_bootstrap' in E else None)
                                                                                for k, E in R['evals'].items()},
                                                              permutation={str(k): {s: {kk: vv for kk, vv in v[s].items()
                                                                                        if kk in ('observed_stat', 'p_cell_shuffle', 'p_toroidal_shift', 'q95_shuffle', 'q95_shift')}
                                                                                    for s in v} for k, v in R['perms'].items()})
            if wdw['is_calibration']:
                calib = dict(F=F, R=R)

    # Fig. 2 and sigma_b figure for the first two windows
    labs = list(fields.keys())
    if len(labs) >= 2:
        F1, F2 = fields[labs[0]], fields[labs[1]]
        e1 = F1['events'][F1['events']['magnitude'] >= 5.0]; e2 = F2['events'][F2['events']['magnitude'] >= 5.0]
        viz.plot_fig2(F1['grid'], F2['grid'], F1['label'], F2['label'], cfg.grid_res, os.path.join(out, 'Fig2_b_mu_maps.pdf'), e1, e2)
        viz.plot_fig_bsigma(F1['grid'], F2['grid'], F1['label'], F2['label'], cfg.grid_res, os.path.join(out, 'Fig_b_sigma.pdf'), cfg.sigma_b_mask)

    # ------------------------------------------------------------------ stage 4: forward
    for fw in cfg.forward_predictions:
        if calib is None:
            print("no calibration window -> forward forecast skipped"); break
        Ff = get_fields(fw['train_start'], fw['train_end'])
        thr = calib['R']['thresholds']
        g = Ff['grid'].copy(); b, mu = clean_fields(g)
        g['alarm_b'] = (b <= thr['b_thr']).astype(int); g['alarm_mu'] = (mu >= thr['mu_thr']).astype(int)
        g['alarm_combined'] = (g['alarm_b'] & g['alarm_mu']).astype(int)
        sub = os.path.join(out, f"forecast_{fw['name']}"); os.makedirs(sub, exist_ok=True)
        g.to_csv(os.path.join(sub, f"alarms_{fw['name']}.csv"), index=False)
        try:
            save_three_alarm_shapefiles(g, grid_res=cfg.grid_res, out_basename=os.path.join(sub, f"alarms_{fw['name']}"),
                                        b_threshold=thr['b_thr'], mu_threshold=thr['mu_thr'])
        except Exception as e:
            print(f"  shapefile export skipped: {e}")
        tau_b = float(w[g['alarm_b'] == 1].sum()); tau_mu = float(w[g['alarm_mu'] == 1].sum()); tau_j = float(w[g['alarm_combined'] == 1].sum())
        fsum = dict(name=fw['name'], fields_window=Ff['label'], forecast_window=[fw['forecast_start'].isoformat(), fw['forecast_end'].isoformat()],
                    thresholds_frozen_from=thr, tau_b=tau_b, tau_mu=tau_mu, tau_joint=tau_j,
                    n_cells_joint=int(g['alarm_combined'].sum()), note='thresholds applied unchanged; forecast is unvalidated')
        dump(fsum, os.path.join(sub, f"summary_{fw['name']}.json"))
        print(f"\nFORWARD {fw['name']}: fields {Ff['label']}, frozen b<={thr['b_thr']:.3f}, mu>={thr['mu_thr']:.4g} -> "
              f"tau_b={tau_b:.3f} tau_mu={tau_mu:.3f} tau_joint={tau_j:.3f}")
        Mcal = cfg.calibration_magnitude
        ev = calib['R']['evals'][Mcal]['events']
        viz.plot_fig6(calib['R']['alarm_grid'], g, thr['b_thr'], thr['mu_thr'], cfg.grid_res,
                      f"{calib['F']['label']} -> {calib['R']['tlab']}", fw['name'], ev, os.path.join(out, f"Fig6_alarms_{fw['name']}.pdf"))
        summary.setdefault('forward', {})[fw['name']] = {k: v for k, v in fsum.items() if k != 'thresholds_frozen_from'}

    dump(summary, os.path.join(out, 'run_summary.json'))
    print(f"\n{'=' * 70}\nDONE - results in {out}\n{'=' * 70}")


if __name__ == '__main__':
    main()
