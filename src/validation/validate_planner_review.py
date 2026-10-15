
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "approval"))

import purchase_order_review as review


# ============================================================
# PHASE 11B.2 - PLANNER REVIEW VALIDATION
# ============================================================

passed = 0


def check(name, condition):
    global passed

    if not condition:
        raise AssertionError(name)

    passed += 1
    print(f"[PASS] {name}")


def expect_error(name, function):
    try:
        function()
    except ValueError:
        check(name, True)
        return

    raise AssertionError(name)


def run_validation():
    print("=" * 65)
    print("PHASE 11B.2 - PLANNER REVIEW VALIDATION")
    print("=" * 65)

    # Use a temporary database so production data is untouched.
    with tempfile.TemporaryDirectory() as temp_dir:

        test_db = Path(temp_dir) / "test_reviews.db"

        with patch.object(review, "DB_PATH", test_db):

            sync = review.sync_proposals()

            check(
                "Test proposals imported",
                sync["source_proposals"] == 993
                and sync["inserted"] == 993,
            )

            register = review.get_review_register()
            original = register.iloc[0]

            proposal_id = original["proposal_id"]
            original_qty = int(original["original_qty"])
            original_value = float(original["original_value"])

            new_qty = original_qty + 10

            result = review.update_planner_review(
                proposal_id=proposal_id,
                reviewed_qty=new_qty,
                planner_notes="Adjusted after planner review",
                changed_by="TEST_PLANNER",
            )

            check(
                "Planner update saved",
                result["updated"] is True,
            )

            updated = review.get_review_register()

            row = updated.loc[
                updated["proposal_id"] == proposal_id
            ].iloc[0]

            check(
                "Reviewed quantity updated",
                int(row["reviewed_qty"]) == new_qty,
            )

            check(
                "Original quantity protected",
                int(row["original_qty"]) == original_qty,
            )

            check(
                "Original value protected",
                abs(
                    float(row["original_value"]) - original_value
                ) < 0.01,
            )

            expected_value = round(
                new_qty * original_value / original_qty,
                2,
            )

            check(
                "Reviewed value recalculated",
                abs(
                    float(row["reviewed_value"]) - expected_value
                ) < 0.01,
            )

            check(
                "Planner notes saved",
                row["planner_notes"]
                == "Adjusted after planner review",
            )

            audit = review.get_proposal_audit(proposal_id)

            check(
                "Quantity change audited",
                "QUANTITY_UPDATED" in audit["action"].values,
            )

            check(
                "Notes change audited",
                "NOTES_UPDATED" in audit["action"].values,
            )

            changes = audit[
                audit["action"].isin([
                    "QUANTITY_UPDATED",
                    "NOTES_UPDATED",
                ])
            ]

            check(
                "Planner identity recorded",
                (changes["changed_by"] == "TEST_PLANNER").all(),
            )

            audit_before = len(audit)

            repeated = review.update_planner_review(
                proposal_id,
                new_qty,
                "Adjusted after planner review",
                "TEST_PLANNER",
            )

            check(
                "Identical update ignored",
                repeated["updated"] is False,
            )

            check(
                "No duplicate audit events",
                len(review.get_proposal_audit(proposal_id))
                == audit_before,
            )

            expect_error(
                "Zero quantity rejected",
                lambda: review.update_planner_review(
                    proposal_id, 0, "", "TEST_PLANNER"
                ),
            )

            expect_error(
                "Negative quantity rejected",
                lambda: review.update_planner_review(
                    proposal_id, -5, "", "TEST_PLANNER"
                ),
            )

            expect_error(
                "Missing planner rejected",
                lambda: review.update_planner_review(
                    proposal_id, 10, "", ""
                ),
            )

            expect_error(
                "Unknown proposal rejected",
                lambda: review.update_planner_review(
                    "UNKNOWN-ID", 10, "", "TEST_PLANNER"
                ),
            )

            # Simulate an approved proposal.
            with review.get_connection() as conn:
                conn.execute(
                    """
                    UPDATE purchase_order_reviews
                    SET status = 'Approved'
                    WHERE proposal_id = ?
                    """,
                    (proposal_id,),
                )

            expect_error(
                "Approved proposal locked",
                lambda: review.update_planner_review(
                    proposal_id, 20, "", "TEST_PLANNER"
                ),
            )

            check(
                "Temporary database isolated",
                review.DB_PATH == test_db,
            )

    print("\n" + "=" * 65)
    print(f"PASSED: {passed}")
    print("FAILED: 0")
    print("PHASE 11B.2 VALIDATION PASSED")
    print("=" * 65)


if __name__ == "__main__":
    run_validation()
