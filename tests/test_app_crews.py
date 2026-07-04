# ============================================================
# tests/test_app_crews.py
# ============================================================
# WHY: Verify custom app module exposes expected callable functions.

import pandas as pd
from sklearn.linear_model import LinearRegression

from mlstudio import app_crews


def test_app_crews_has_main() -> None:
    """Verify the custom module exposes a main function."""
    assert callable(app_crews.main)


def test_report_top_driver_runs() -> None:
    """Verify top-driver reporting runs with a trained model."""
    x = pd.DataFrame(
        {
            "hours_studied": [1.0, 2.0, 3.0, 4.0],
            "practice_quizzes": [1, 1, 2, 3],
            "attendance_pct": [80, 82, 90, 95],
            "sleep_hours": [6.0, 6.5, 7.0, 7.5],
            "prior_score": [60, 65, 70, 80],
        }
    )
    y = pd.Series([58.0, 64.0, 72.0, 85.0])

    model = LinearRegression()
    model.fit(x, y)

    app_crews.report_top_driver(model)
