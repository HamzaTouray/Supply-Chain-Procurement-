from pathlib import Path
from contextlib import contextmanager
import hashlib
import math
import sqlite3
import pandas as pd

# Phase 11C.1 - Purchase Order Review and Approval Backend
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'processed'
DB_PATH = ROOT / 'data' / 'purchase_order_reviews.db'
ACTIONS_PATH = DATA_DIR / 'feasible_supply_actions.csv'
REQUIRED_COLUMNS = ['sku', 'supplier_id', 'recommended_order_qty', 'recommended_order_value']
STATUSES = ('Draft', 'Pending Approval', 'Approved', 'Rejected')

@contextmanager
def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        connection.execute('PRAGMA foreign_keys = ON')
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS purchase_order_reviews (
                proposal_id TEXT PRIMARY KEY,
                source_row_number INTEGER NOT NULL,
                sku TEXT NOT NULL,
                supplier_id TEXT NOT NULL,
                original_qty REAL NOT NULL,
                original_value REAL NOT NULL,
                reviewed_qty REAL NOT NULL,
                reviewed_value REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'Draft'
                    CHECK (status IN ('Draft', 'Pending Approval', 'Approved', 'Rejected')),
                planner_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS purchase_order_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                changed_by TEXT,
                changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (proposal_id) REFERENCES purchase_order_reviews(proposal_id)
            )
        ''')


def load_proposals():
    if not ACTIONS_PATH.exists():
        raise FileNotFoundError(f'Planning file not found: {ACTIONS_PATH}')
    df = pd.read_csv(ACTIONS_PATH)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if df.empty:
        raise ValueError('Planning proposal file is empty.')
    return df


def build_proposal_id(row):
    source_values = '|'.join(str(value) for value in row.tolist())
    digest = hashlib.sha256(source_values.encode('utf-8')).hexdigest()[:16]
    return f'PO-{digest}'


def sync_proposals():
    initialize_database()
    proposals = load_proposals().fillna('')
    inserted = 0
    existing = 0
    occurrence_counts = {}
    with get_connection() as conn:
        for row_number, (_, row) in enumerate(proposals.iterrows(), start=1):
            base_id = build_proposal_id(row)
            occurrence_counts[base_id] = occurrence_counts.get(base_id, 0) + 1
            proposal_id = f'{base_id}-{occurrence_counts[base_id]:03d}'
            sku = str(row['sku'])
            supplier = str(row['supplier_id'])
            qty = float(row['recommended_order_qty'])
            value = float(row['recommended_order_value'])
            if not math.isfinite(qty) or not math.isfinite(value) or qty <= 0 or value < 0:
                raise ValueError(f'Invalid proposal: {proposal_id}')
            cursor = conn.execute('''
                INSERT OR IGNORE INTO purchase_order_reviews (
                    proposal_id, source_row_number, sku, supplier_id,
                    original_qty, original_value, reviewed_qty, reviewed_value, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Draft')
            ''', (proposal_id, row_number, sku, supplier, qty, value, qty, value))
            if cursor.rowcount == 1:
                inserted += 1
                conn.execute('''
                    INSERT INTO purchase_order_audit
                    (proposal_id, action, old_value, new_value, changed_by)
                    VALUES (?, 'CREATED', NULL, 'Draft', 'SYSTEM')
                ''', (proposal_id,))
            else:
                existing += 1
    return {'source_proposals': len(proposals), 'inserted': inserted, 'existing': existing}


def get_review_register():
    initialize_database()
    with get_connection() as conn:
        return pd.read_sql_query('''
            SELECT proposal_id, source_row_number, sku, supplier_id,
                   original_qty, original_value, reviewed_qty, reviewed_value,
                   status, planner_notes, created_at, updated_at
            FROM purchase_order_reviews ORDER BY source_row_number
        ''', conn)


def validate_register():
    source = load_proposals()
    register = get_review_register()
    if len(source) != len(register):
        raise AssertionError('Proposal count does not match source.')
    if not register['proposal_id'].is_unique:
        raise AssertionError('Duplicate proposal IDs detected.')
    if not register['status'].isin(STATUSES).all():
        raise AssertionError('Unexpected approval statuses.')
    source_qty = pd.to_numeric(source['recommended_order_qty']).sum()
    registered_qty = register['original_qty'].sum()
    if abs(source_qty - registered_qty) > 0.001:
        raise AssertionError('Original quantities do not reconcile.')
    source_value = pd.to_numeric(source['recommended_order_value']).sum()
    registered_value = register['original_value'].sum()
    if abs(source_value - registered_value) > 0.01:
        raise AssertionError('Original purchasing values do not reconcile.')
    with get_connection() as conn:
        audit_count = conn.execute('''
            SELECT COUNT(*) FROM purchase_order_audit WHERE action = 'CREATED'
        ''').fetchone()[0]
    if audit_count != len(register):
        raise AssertionError('CREATED audit record count does not match.')
    return {'proposal_count': len(register), 'total_units': registered_qty,
            'total_value': registered_value, 'audit_records': audit_count}


def _require_actor(actor, label):
    if actor is None or not str(actor).strip():
        raise ValueError(f'{label} is required.')
    actor = str(actor).strip()
    if len(actor) > 200:
        raise ValueError(f'{label} cannot exceed 200 characters.')
    return actor


def update_planner_review(proposal_id, reviewed_qty, planner_notes, changed_by):
    proposal_id = str(proposal_id).strip()
    changed_by = _require_actor(changed_by, 'Planner identifier')
    if not proposal_id:
        raise ValueError('Proposal ID is required.')
    if reviewed_qty is None or isinstance(reviewed_qty, bool):
        raise ValueError('Reviewed quantity must be a positive number.')
    try:
        reviewed_qty = float(reviewed_qty)
    except (TypeError, ValueError) as exc:
        raise ValueError('Reviewed quantity must be numeric.') from exc
    if not math.isfinite(reviewed_qty) or reviewed_qty <= 0:
        raise ValueError('Reviewed quantity must be greater than zero.')
    if not reviewed_qty.is_integer():
        raise ValueError('Reviewed quantity must be a whole number.')
    reviewed_qty = int(reviewed_qty)
    planner_notes = '' if planner_notes is None else str(planner_notes).strip()
    if len(planner_notes) > 2000:
        raise ValueError('Planner notes cannot exceed 2000 characters.')
    initialize_database()
    with get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        record = conn.execute('''
            SELECT original_qty, original_value, reviewed_qty, reviewed_value,
                   planner_notes, status FROM purchase_order_reviews WHERE proposal_id = ?
        ''', (proposal_id,)).fetchone()
        if record is None:
            raise ValueError(f'Proposal not found: {proposal_id}')
        original_qty, original_value, old_qty, old_value, old_notes, status = record
        if status != 'Draft':
            raise ValueError(f'Cannot edit proposal {proposal_id}. Current status: {status}')
        if original_qty <= 0:
            raise ValueError('Original proposal quantity is invalid.')
        unit_cost = original_value / original_qty
        reviewed_value = round(reviewed_qty * unit_cost, 2)
        qty_changed = abs(float(old_qty) - reviewed_qty) > 0.000001
        notes_changed = old_notes != planner_notes
        if not qty_changed and not notes_changed:
            return {'proposal_id': proposal_id, 'updated': False,
                    'reviewed_qty': reviewed_qty, 'reviewed_value': reviewed_value,
                    'status': status}
        conn.execute('''
            UPDATE purchase_order_reviews SET reviewed_qty = ?, reviewed_value = ?,
            planner_notes = ?, updated_at = CURRENT_TIMESTAMP WHERE proposal_id = ?
        ''', (reviewed_qty, reviewed_value, planner_notes, proposal_id))
        if qty_changed:
            conn.execute('''
                INSERT INTO purchase_order_audit
                (proposal_id, action, old_value, new_value, changed_by)
                VALUES (?, 'QUANTITY_UPDATED', ?, ?, ?)
            ''', (proposal_id, str(int(old_qty)), str(reviewed_qty), changed_by))
        if notes_changed:
            conn.execute('''
                INSERT INTO purchase_order_audit
                (proposal_id, action, old_value, new_value, changed_by)
                VALUES (?, 'NOTES_UPDATED', ?, ?, ?)
            ''', (proposal_id, old_notes, planner_notes, changed_by))
    return {'proposal_id': proposal_id, 'updated': True,
            'reviewed_qty': reviewed_qty, 'reviewed_value': reviewed_value,
            'status': 'Draft'}


def get_proposal_audit(proposal_id):
    initialize_database()
    with get_connection() as conn:
        return pd.read_sql_query('''
            SELECT audit_id, proposal_id, action, old_value, new_value,
                   changed_by, changed_at FROM purchase_order_audit
            WHERE proposal_id = ? ORDER BY audit_id
        ''', conn, params=(proposal_id,))


# Phase 11C.1 - Controlled approval status transitions.
# Actor IDs are recorded for auditing; identity/role authentication is not implemented.
def _transition_status(proposal_id, actor, expected_status, new_status, action,
                       reason=None):
    proposal_id = str(proposal_id).strip()
    actor = _require_actor(actor, 'Actor identifier')
    if not proposal_id:
        raise ValueError('Proposal ID is required.')
    if reason is not None:
        reason = str(reason).strip()
        if len(reason) > 2000:
            raise ValueError('Decision reason cannot exceed 2000 characters.')
    if new_status == 'Rejected' and not reason:
        raise ValueError('A rejection reason is required.')
    initialize_database()
    with get_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('''
            SELECT status, reviewed_qty, reviewed_value FROM purchase_order_reviews
            WHERE proposal_id = ?
        ''', (proposal_id,)).fetchone()
        if row is None:
            raise ValueError(f'Proposal not found: {proposal_id}')
        old_status, reviewed_qty, reviewed_value = row
        if old_status != expected_status:
            raise ValueError(
                f'Cannot {action.lower().replace("_", " ")} proposal {proposal_id}. '
                f'Expected {expected_status}; current status: {old_status}'
            )
        if reviewed_qty <= 0 or reviewed_value < 0:
            raise ValueError('Proposal reviewed quantity/value is invalid.')
        conn.execute('''
            UPDATE purchase_order_reviews SET status = ?,
            updated_at = CURRENT_TIMESTAMP WHERE proposal_id = ?
        ''', (new_status, proposal_id))
        conn.execute('''
            INSERT INTO purchase_order_audit
            (proposal_id, action, old_value, new_value, changed_by)
            VALUES (?, ?, ?, ?, ?)
        ''', (proposal_id, action, old_status, new_status, actor))
        if reason:
            conn.execute('''
                INSERT INTO purchase_order_audit
                (proposal_id, action, old_value, new_value, changed_by)
                VALUES (?, 'DECISION_REASON', NULL, ?, ?)
            ''', (proposal_id, reason, actor))
    return {'proposal_id': proposal_id, 'old_status': old_status,
            'status': new_status, 'changed_by': actor, 'reason': reason}


def submit_for_approval(proposal_id, changed_by):
    return _transition_status(proposal_id, changed_by, 'Draft',
                              'Pending Approval', 'SUBMITTED_FOR_APPROVAL')


def approve_proposal(proposal_id, changed_by):
    return _transition_status(proposal_id, changed_by, 'Pending Approval',
                              'Approved', 'APPROVED')


def reject_proposal(proposal_id, changed_by, reason):
    return _transition_status(proposal_id, changed_by, 'Pending Approval',
                              'Rejected', 'REJECTED', reason=reason)


if __name__ == '__main__':
    print('=' * 65)
    print('PHASE 11C.1 - PURCHASE ORDER REVIEW BACKEND')
    print('=' * 65)
    sync = sync_proposals()
    print(f"Source proposals: {sync['source_proposals']:,}")
    print(f"New proposals registered: {sync['inserted']:,}")
    print(f"Existing proposals: {sync['existing']:,}")
    result = validate_register()
    print('\nVALIDATION RESULTS')
    print(f"Proposal count: {result['proposal_count']:,}")
    print(f"Total units: {result['total_units']:,.0f}")
    print(f"Total value: ${result['total_value']:,.2f}")
    print(f"CREATED audit records: {result['audit_records']:,}")
    print('\n[PASS] Purchase order review register validated')
