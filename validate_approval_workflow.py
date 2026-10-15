
"""Phase 11C.2 - Isolated approval workflow validation."""

from pathlib import Path
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "approval"))

import purchase_order_review as review


passed = 0
failed = 0


def check(label, condition):
    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {label}")
    else:
        failed += 1
        print(f"[FAIL] {label}")


def rejects(label, function):
    try:
        function()
    except ValueError:
        check(label, True)
    except Exception as exc:
        check(label, False)
        print(
            f"       Unexpected error: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        check(label, False)


def state(proposal_id):
    with review.get_connection() as conn:
        return conn.execute(
            """
            SELECT
                status,
                original_qty,
                original_value,
                reviewed_qty,
                reviewed_value
            FROM purchase_order_reviews
            WHERE proposal_id = ?
            """,
            (proposal_id,),
        ).fetchone()


def actions(proposal_id):
    audit = review.get_proposal_audit(proposal_id)
    return audit["action"].tolist()


def main():
    original_db = review.DB_PATH
    original_actions = review.ACTIONS_PATH

    try:
        with tempfile.TemporaryDirectory(
            prefix="phase11c2_"
        ) as directory:

            base = Path(directory)

            review.DB_PATH = base / "test_reviews.db"
            review.ACTIONS_PATH = base / "test_actions.csv"

            test_data = pd.DataFrame(
                [
                    {
                        "sku": "TEST-A",
                        "supplier_id": "SUP-01",
                        "recommended_order_qty": 10,
                        "recommended_order_value": 250,
                    },
                    {
                        "sku": "TEST-B",
                        "supplier_id": "SUP-02",
                        "recommended_order_qty": 8,
                        "recommended_order_value": 160,
                    },
                    {
                        "sku": "TEST-C",
                        "supplier_id": "SUP-03",
                        "recommended_order_qty": 4,
                        "recommended_order_value": 40,
                    },
                ]
            )

            test_data.to_csv(
                review.ACTIONS_PATH,
                index=False,
            )

            print("=" * 65)
            print(
                "PHASE 11C.2 - APPROVAL WORKFLOW VALIDATION"
            )
            print("=" * 65)

            synced = review.sync_proposals()

            check(
                "Three test proposals imported",
                synced["inserted"] == 3,
            )

            check(
                "Temporary database created",
                review.DB_PATH.exists(),
            )

            register = review.get_review_register()

            check(
                "All proposals start as Draft",
                len(register) == 3
                and register["status"].eq("Draft").all(),
            )

            a, b, c = register["proposal_id"].tolist()

            # --------------------------------------------
            # Planner review
            # --------------------------------------------

            edited = review.update_planner_review(
                a,
                12,
                "Adjusted for test",
                "planner-test",
            )

            check(
                "Draft quantity and notes updated",
                edited["updated"]
                and state(a)[3] == 12,
            )

            check(
                "Reviewed value recalculated",
                abs(state(a)[4] - 300) < 0.001,
            )

            # --------------------------------------------
            # Submit for approval
            # --------------------------------------------

            submitted = review.submit_for_approval(
                a,
                "planner-test",
            )

            check(
                "Draft submitted to Pending Approval",
                submitted["status"] == "Pending Approval"
                and state(a)[0] == "Pending Approval",
            )

            audit_a = review.get_proposal_audit(a)

            check(
                "Submission actor audited",
                "SUBMITTED_FOR_APPROVAL" in actions(a)
                and "planner-test"
                in audit_a["changed_by"].tolist(),
            )

            rejects(
                "Submitted proposal cannot be edited",
                lambda: review.update_planner_review(
                    a,
                    15,
                    "Edit after submission",
                    "planner-test",
                ),
            )

            rejects(
                "Cannot submit same proposal twice",
                lambda: review.submit_for_approval(
                    a,
                    "planner-test",
                ),
            )

            rejects(
                "Cannot reject without reason",
                lambda: review.reject_proposal(
                    a,
                    "approver-test",
                    "",
                ),
            )

            check(
                "Failed rejection preserves status",
                state(a)[0] == "Pending Approval",
            )

            # --------------------------------------------
            # Approve proposal
            # --------------------------------------------

            approved = review.approve_proposal(
                a,
                "approver-test",
            )

            check(
                "Pending proposal approved",
                approved["status"] == "Approved"
                and state(a)[0] == "Approved",
            )

            check(
                "Approval actor audited",
                "APPROVED" in actions(a)
                and "approver-test"
                in review.get_proposal_audit(a)[
                    "changed_by"
                ].tolist(),
            )

            rejects(
                "Approved proposal cannot be edited",
                lambda: review.update_planner_review(
                    a,
                    12,
                    "Change",
                    "planner-test",
                ),
            )

            rejects(
                "Approved proposal cannot be approved twice",
                lambda: review.approve_proposal(
                    a,
                    "approver-test",
                ),
            )

            rejects(
                "Approved proposal cannot be rejected",
                lambda: review.reject_proposal(
                    a,
                    "approver-test",
                    "Too late",
                ),
            )

            check(
                "Original quantity and value protected",
                state(a)[1:3] == (10, 250),
            )

            check(
                "Reviewed quantity and value preserved",
                state(a)[3:] == (12, 300),
            )

            # --------------------------------------------
            # Reject proposal
            # --------------------------------------------

            review.submit_for_approval(
                b,
                "planner-test",
            )

            rejected = review.reject_proposal(
                b,
                "approver-test",
                "Supplier capacity concern",
            )

            check(
                "Pending proposal rejected",
                rejected["status"] == "Rejected"
                and state(b)[0] == "Rejected",
            )

            audit_b = review.get_proposal_audit(b)

            check(
                "Rejection reason audited",
                "DECISION_REASON" in actions(b)
                and "Supplier capacity concern"
                in audit_b["new_value"].tolist(),
            )

            rejects(
                "Rejected proposal cannot be edited",
                lambda: review.update_planner_review(
                    b,
                    9,
                    "Change",
                    "planner-test",
                ),
            )

            rejects(
                "Rejected proposal cannot be approved",
                lambda: review.approve_proposal(
                    b,
                    "approver-test",
                ),
            )

            rejects(
                "Rejected proposal cannot be resubmitted",
                lambda: review.submit_for_approval(
                    b,
                    "planner-test",
                ),
            )

            # --------------------------------------------
            # Invalid transitions
            # --------------------------------------------

            rejects(
                "Draft cannot be approved directly",
                lambda: review.approve_proposal(
                    c,
                    "approver-test",
                ),
            )

            rejects(
                "Draft cannot be rejected directly",
                lambda: review.reject_proposal(
                    c,
                    "approver-test",
                    "Reason",
                ),
            )

            rejects(
                "Blank planner identifier rejected",
                lambda: review.submit_for_approval(
                    c,
                    "  ",
                ),
            )

            rejects(
                "Blank approver identifier rejected",
                lambda: review.approve_proposal(
                    c,
                    "  ",
                ),
            )

            rejects(
                "Unknown proposal rejected",
                lambda: review.submit_for_approval(
                    "UNKNOWN",
                    "planner-test",
                ),
            )

            check(
                "Third proposal remains Draft",
                state(c)[0] == "Draft",
            )

            # --------------------------------------------
            # Audit and data integrity
            # --------------------------------------------

            check(
                "Status changes audited exactly once",
                actions(a).count(
                    "SUBMITTED_FOR_APPROVAL"
                ) == 1
                and actions(a).count("APPROVED") == 1
                and actions(b).count(
                    "SUBMITTED_FOR_APPROVAL"
                ) == 1
                and actions(b).count("REJECTED") == 1,
            )

            check(
                "Rejection reason recorded once",
                actions(b).count(
                    "DECISION_REASON"
                ) == 1,
            )

            sync_again = review.sync_proposals()

            check(
                "Resync does not duplicate proposals",
                sync_again["inserted"] == 0
                and sync_again["existing"] == 3,
            )

            check(
                "Resync preserves approval statuses",
                [
                    state(i)[0]
                    for i in (a, b, c)
                ] == [
                    "Approved",
                    "Rejected",
                    "Draft",
                ],
            )

            check(
                "Register validation accepts statuses",
                review.validate_register()[
                    "proposal_count"
                ] == 3,
            )

            check(
                "Real database not used",
                review.DB_PATH != original_db
                and review.ACTIONS_PATH != original_actions,
            )

    finally:
        review.DB_PATH = original_db
        review.ACTIONS_PATH = original_actions

    print()
    print("=" * 65)
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")

    if failed:
        print("PHASE 11C.2 VALIDATION FAILED")
        raise SystemExit(1)

    print("PHASE 11C.2 VALIDATION PASSED")


if __name__ == "__main__":
    main()
