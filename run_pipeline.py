# ─────────────────────────────────────────────────────────────
#  EcoGrid — run_pipeline.py
#  Run ONCE to train all models and save artefacts.
#
#  Usage:
#    python run_pipeline.py --data egrid2023_data_rev2.xlsx
# ─────────────────────────────────────────────────────────────

import argparse, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.pipeline import run_pipeline
from utils.train    import split_data, train_all, save_artefacts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='egrid2023_data_rev2.xlsx')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] File not found: {args.data}")
        print("  Download from: https://www.epa.gov/egrid")
        sys.exit(1)

    # Phases 02-06
    df = run_pipeline(args.data)

    # Phases 07-08
    X_train, X_test, y_train, y_test = split_data(df)
    trained, best_name = train_all(X_train, y_train, X_test, y_test)
    save_artefacts(trained, df)

    print(f"\n✅  Done. Best model: {best_name}")
    print("    Now run:  streamlit run app.py")


if __name__ == '__main__':
    main()
