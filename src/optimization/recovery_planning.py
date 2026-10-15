from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'processed'
INPUTS = {
    'actions': 'feasible_supply_actions.csv',
    'exceptions': 'supply_exceptions.csv',
    'suppliers': 'supplier_execution_summary.csv',
    'projection': 'feasible_inventory_projection.csv',
    'open_pos': 'open_purchase_orders_clean.csv',
    'summary': 'time_phased_planning_summary.csv',
}
TOL = 1e-7


def read_inputs():
    print('\n' + '=' * 90)
    print('PHASE 7G - SUPPLY RECOVERY AND ACTION PRIORITIZATION')
    print('=' * 90)
    data = {}
    for key, filename in INPUTS.items():
        path = DATA / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing input: {path}')
        data[key] = pd.read_csv(path)
        print(f'Loaded {filename}: {len(data[key]):,} rows')
    required = {
        'actions': ['sku', 'supplier_id', 'required_receipt_week', 'earliest_feasible_receipt', 'receipt_delay_days', 'recommended_order_qty', 'recommended_order_value', 'execution_action'],
        'projection': ['sku', 'forecast_week', 'beginning_inventory', 'confirmed_existing_receipts', 'feasible_new_receipts', 'forecast_demand', 'ending_inventory', 'safety_stock', 'below_safety_stock', 'stockout'],
        'open_pos': ['po_number', 'sku', 'supplier_id', 'open_qty', 'expected_date', 'overdue_flag', 'partial_receipt_flag', 'expected_after_promise_flag'],
        'summary': ['sku', 'supplier_id', 'unit_cost', 'safety_stock'],
        'exceptions': ['sku', 'exception_type'],
        'suppliers': ['supplier_id'],
    }
    for key, columns in required.items():
        missing = set(columns) - set(data[key].columns)
        if missing:
            raise ValueError(f'{key} missing columns: {sorted(missing)}')
    if not data['summary']['sku'].is_unique:
        raise ValueError('SKU summary contains duplicate SKUs')
    if len(data['projection']) != 5200 or data['projection']['sku'].nunique() != 200:
        raise ValueError('Expected 5,200 projection rows and 200 SKUs')
    if len(data['actions']) != 993:
        raise ValueError('Expected 993 baseline consolidated order lines')
    return data


def flags(series):
    return series.astype(str).str.strip().str.lower().isin(['true', '1', 'yes'])


def prepare(data):
    a = data['actions'].copy()
    p = data['projection'].copy()
    po = data['open_pos'].copy()
    m = data['summary'].copy()
    for col in ['required_receipt_week', 'earliest_feasible_receipt']:
        a[col] = pd.to_datetime(a[col], errors='coerce').dt.normalize()
    p['forecast_week'] = pd.to_datetime(p['forecast_week'], errors='coerce').dt.normalize()
    po['expected_date'] = pd.to_datetime(po['expected_date'], errors='coerce').dt.normalize()
    for col in ['overdue_flag', 'partial_receipt_flag', 'expected_after_promise_flag']:
        po[col] = flags(po[col])
    p['below_safety_stock'] = flags(p['below_safety_stock'])
    p['stockout'] = flags(p['stockout'])
    numeric = ['beginning_inventory', 'confirmed_existing_receipts', 'feasible_new_receipts', 'forecast_demand', 'ending_inventory', 'safety_stock']
    for col in numeric:
        p[col] = pd.to_numeric(p[col], errors='raise')
    for col in ['recommended_order_qty', 'recommended_order_value', 'receipt_delay_days']:
        a[col] = pd.to_numeric(a[col], errors='raise')
    po['open_qty'] = pd.to_numeric(po['open_qty'], errors='raise')
    if a[['required_receipt_week', 'earliest_feasible_receipt']].isna().any().any() or p['forecast_week'].isna().any():
        raise ValueError('Invalid order or projection dates')
    if p[['sku', 'forecast_week']].duplicated().any():
        raise ValueError('Duplicate SKU/week projection records')
    if not set(p['sku']).issubset(set(m['sku'])):
        raise ValueError('Projection SKUs not found in master summary')
    return a, p, po, m


def build_shortage_windows(projection, summary):
    print('\nAnalyzing feasible inventory shortages...')
    p = projection.sort_values(['sku', 'forecast_week']).copy()
    p['safety_stock_gap_units'] = (p['safety_stock'] - p['ending_inventory']).clip(lower=0)
    p['unmet_demand_proxy_units'] = (-p['ending_inventory']).clip(lower=0)
    p['new_shortage_exposure_units'] = p.groupby('sku')['unmet_demand_proxy_units'].diff().fillna(p['unmet_demand_proxy_units']).clip(lower=0)
    # Negative projected inventory is a backlog/exposure proxy, not verified lost sales.
    windows = []
    for sku, g in p.groupby('sku', sort=False):
        g = g.sort_values('forecast_week')
        active = None
        for row in g.itertuples(index=False):
            if row.below_safety_stock:
                if active is None:
                    active = {'sku': sku, 'first_breach_week': row.forecast_week, 'last_breach_week': row.forecast_week,
                              'breach_weeks': 0, 'stockout_weeks': 0, 'peak_safety_stock_gap_units': 0., 'peak_negative_inventory_units': 0.}
                active['last_breach_week'] = row.forecast_week
                active['breach_weeks'] += 1
                active['stockout_weeks'] += int(row.stockout)
                active['peak_safety_stock_gap_units'] = max(active['peak_safety_stock_gap_units'], float(row.safety_stock_gap_units))
                active['peak_negative_inventory_units'] = max(active['peak_negative_inventory_units'], float(row.unmet_demand_proxy_units))
            elif active is not None:
                windows.append(active)
                active = None
        if active is not None:
            windows.append(active)
    window_df = pd.DataFrame(windows, columns=['sku', 'first_breach_week', 'last_breach_week', 'breach_weeks', 'stockout_weeks', 'peak_safety_stock_gap_units', 'peak_negative_inventory_units'])
    if not window_df.empty:
        window_df = window_df.merge(summary[['sku', 'supplier_id', 'unit_cost']], on='sku', how='left', validate='many_to_one')
        window_df['peak_shortage_exposure_value'] = window_df['peak_negative_inventory_units'] * window_df['unit_cost']
    return p, window_df


def build_recovery_actions(actions, projection, open_pos, summary):
    print('Prioritizing SKU recovery actions...')
    exposure = projection.groupby('sku', as_index=False).agg(
        first_breach_week=('forecast_week', lambda x: x.loc[projection.loc[x.index, 'below_safety_stock']].min()),
        first_stockout_week=('forecast_week', lambda x: x.loc[projection.loc[x.index, 'stockout']].min()),
        safety_breach_weeks=('below_safety_stock', 'sum'),
        stockout_weeks=('stockout', 'sum'),
        peak_safety_stock_gap_units=('safety_stock_gap_units', 'max'),
        peak_negative_inventory_units=('unmet_demand_proxy_units', 'max'),
    )
    orders = actions.groupby('sku', as_index=False).agg(
        new_order_lines=('sku', 'size'),
        new_order_units=('recommended_order_qty', 'sum'),
        new_order_value=('recommended_order_value', 'sum'),
        late_new_order_lines=('receipt_delay_days', lambda x: int((x > 0).sum())),
        max_new_order_delay_days=('receipt_delay_days', 'max'),
        first_required_receipt=('required_receipt_week', 'min'),
        first_feasible_receipt=('earliest_feasible_receipt', 'min'),
    )
    po = open_pos.copy()
    po['uncertain'] = po[['overdue_flag', 'partial_receipt_flag', 'expected_after_promise_flag']].any(axis=1)
    # A historical expected date also makes an open PO uncertain.
    snapshot = projection['forecast_week'].min()
    po['uncertain'] = po['uncertain'] | (po['expected_date'] < snapshot)
    po['uncertain_units'] = np.where(po['uncertain'], po['open_qty'], 0)
    po_summary = po.groupby('sku', as_index=False).agg(
        existing_open_po_lines=('sku', 'size'),
        uncertain_existing_po_lines=('uncertain', 'sum'),
        uncertain_existing_po_units=('uncertain_units', 'sum'),
    )
    result = summary[['sku', 'supplier_id', 'unit_cost']].merge(exposure, on='sku', how='left', validate='one_to_one')
    result = result.merge(orders, on='sku', how='left', validate='one_to_one').merge(po_summary, on='sku', how='left', validate='one_to_one')
    numeric_cols = ['safety_breach_weeks', 'stockout_weeks', 'peak_safety_stock_gap_units', 'peak_negative_inventory_units', 'new_order_lines', 'new_order_units', 'new_order_value', 'late_new_order_lines', 'max_new_order_delay_days', 'existing_open_po_lines', 'uncertain_existing_po_lines', 'uncertain_existing_po_units']
    result[numeric_cols] = result[numeric_cols].fillna(0)
    result['peak_shortage_exposure_value'] = result['peak_negative_inventory_units'] * result['unit_cost']
    result['priority'] = np.select(
        [(result['stockout_weeks'] > 0) & ((result['uncertain_existing_po_units'] > 0) | (result['late_new_order_lines'] > 0)),
         result['stockout_weeks'] > 0,
         result['safety_breach_weeks'] > 0,
         (result['late_new_order_lines'] > 0) | (result['uncertain_existing_po_lines'] > 0)],
        ['CRITICAL', 'HIGH', 'MEDIUM', 'WATCH'], default='NO EXCEPTION')
    result['recommended_action'] = np.select(
        [(result['stockout_weeks'] > 0) & (result['uncertain_existing_po_units'] > 0),
         (result['stockout_weeks'] > 0) & (result['late_new_order_lines'] > 0),
         result['stockout_weeks'] > 0,
         (result['safety_breach_weeks'] > 0) & (result['uncertain_existing_po_units'] > 0),
         result['safety_breach_weeks'] > 0,
         result['late_new_order_lines'] > 0,
         result['uncertain_existing_po_lines'] > 0],
        ['CONFIRM / EXPEDITE OPEN PO; ASSESS ALTERNATE SUPPLY',
         'REQUEST EXPEDITE; ASSESS TRANSFER / ALTERNATE SUPPLY',
         'ASSESS EMERGENCY REPLENISHMENT / CUSTOMER ALLOCATION',
         'CONFIRM EXISTING PO DELIVERY; REVIEW BUFFER',
         'REVIEW INVENTORY BUFFER AND ORDER TIMING',
         'CONFIRM FASTER LEAD TIME / EXPEDITE QUOTE',
         'CONFIRM OPEN PO DELIVERY'], default='MONITOR')
    result['recovery_status'] = np.where(result['stockout_weeks'] > 0, 'UNRESOLVED - SIMULATED STOCKOUT',
                                 np.where(result['safety_breach_weeks'] > 0, 'UNRESOLVED - SAFETY STOCK BREACH', 'NO PROJECTED SHORTAGE'))
    result['proposed_expedite_units'] = np.where(result['stockout_weeks'] > 0, np.ceil(result['peak_negative_inventory_units']), 0).astype(int)
    # Proposed expedite quantity is a planning proxy only; not a new PO recommendation.
    rank = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'WATCH': 3, 'NO EXCEPTION': 4}
    result['_rank'] = result['priority'].map(rank)
    return result.sort_values(['_rank', 'peak_shortage_exposure_value', 'stockout_weeks'], ascending=[True, False, False]).drop(columns='_rank').reset_index(drop=True)


def supplier_recovery_summary(recovery):
    result = recovery.groupby('supplier_id', as_index=False).agg(
        skus=('sku', 'size'),
        critical_skus=('priority', lambda x: int((x == 'CRITICAL').sum())),
        high_skus=('priority', lambda x: int((x == 'HIGH').sum())),
        stockout_skus=('stockout_weeks', lambda x: int((x > 0).sum())),
        total_stockout_sku_weeks=('stockout_weeks', 'sum'),
        total_safety_breach_sku_weeks=('safety_breach_weeks', 'sum'),
        late_order_lines=('late_new_order_lines', 'sum'),
        uncertain_po_lines=('uncertain_existing_po_lines', 'sum'),
        uncertain_po_units=('uncertain_existing_po_units', 'sum'),
        new_order_value=('new_order_value', 'sum'),
        sum_of_sku_peak_shortage_exposure_value=('peak_shortage_exposure_value', 'sum'),
    )
    return result.sort_values(['critical_skus', 'total_stockout_sku_weeks'], ascending=False).reset_index(drop=True)


def validate(data, actions, projection, windows, recovery, suppliers):
    print('\n' + '=' * 90)
    print('PHASE 7G VALIDATION')
    print('=' * 90)
    def check(condition, label):
        if not condition:
            raise AssertionError(label)
        print('✓', label)
    check(len(actions) == 993, 'Original 993 consolidated order lines preserved')
    check(len(projection) == 5200, 'All 5,200 projection weeks preserved')
    check(len(recovery) == 200 and recovery['sku'].is_unique, 'Exactly 200 unique SKU recovery records')
    check(int(recovery['safety_breach_weeks'].sum()) == int(projection['below_safety_stock'].sum()), 'Safety stock breach weeks reconcile')
    check(int(recovery['stockout_weeks'].sum()) == int(projection['stockout'].sum()), 'Stockout weeks reconcile')
    check(int(suppliers['total_stockout_sku_weeks'].sum()) == int(projection['stockout'].sum()), 'Supplier stockout exposure reconciles')
    check(int(windows['breach_weeks'].sum()) == int(projection['below_safety_stock'].sum()), 'Consecutive breach windows reconcile')
    check(np.isclose(recovery['new_order_units'].sum(), actions['recommended_order_qty'].sum()), 'Original purchasing quantities preserved')
    check(np.isclose(recovery['new_order_value'].sum(), actions['recommended_order_value'].sum()), 'Original purchasing value preserved')
    check((recovery['proposed_expedite_units'] >= 0).all(), 'Nonnegative recovery exposure quantities')
    print('All Phase 7G validation checks passed.')


def save(actions, projection, windows, recovery, suppliers):
    outputs = {
        'recovery_action_plan.csv': recovery,
        'recovery_shortage_windows.csv': windows,
        'supplier_recovery_priorities.csv': suppliers,
    }
    print('\nSaving Phase 7G outputs...')
    for filename, frame in outputs.items():
        out = frame.copy()
        for col in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                out[col] = out[col].dt.strftime('%Y-%m-%d')
        out.to_csv(DATA / filename, index=False)
        print('✓', filename)


def main():
    data = read_inputs()
    actions, projection, po, summary = prepare(data)
    enriched_projection, windows = build_shortage_windows(projection, summary)
    recovery = build_recovery_actions(actions, enriched_projection, po, summary)
    suppliers = supplier_recovery_summary(recovery)
    validate(data, actions, enriched_projection, windows, recovery, suppliers)
    save(actions, enriched_projection, windows, recovery, suppliers)
    print('\n' + '=' * 90)
    print('PHASE 7G RESULTS')
    print('=' * 90)
    print(f'Baseline consolidated orders: {len(actions):,}')
    print(f'Baseline purchasing value: ${actions["recommended_order_value"].sum():,.2f}')
    print(f'SKUs with stockout exposure: {(recovery["stockout_weeks"] > 0).sum():,}')
    print(f'Safety-stock breach SKU-weeks: {enriched_projection["below_safety_stock"].sum():,}')
    print(f'Stockout SKU-weeks: {enriched_projection["stockout"].sum():,}')
    print(f'Consecutive shortage windows: {len(windows):,}')
    print('\nPriority distribution:')
    print(recovery['priority'].value_counts().to_string())
    print('\nTop 15 recovery priorities:')
    cols = ['sku', 'supplier_id', 'priority', 'stockout_weeks', 'peak_negative_inventory_units', 'peak_shortage_exposure_value', 'recommended_action']
    print(recovery[cols].head(15).to_string(index=False))
    print('\nTop supplier recovery priorities:')
    print(suppliers.head(10).to_string(index=False))
    print('\nPHASE 7G COMPLETE - recovery actions are recommendations, not confirmed expedited supply.')


if __name__ == '__main__':
    main()
