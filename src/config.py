"""Central configuration: paths, random seed, column names and business assumptions."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_CSV = DATA_RAW_DIR / "telco_customer_churn.csv"
RAW_METADATA = DATA_RAW_DIR / "download_metadata.json"
CLEAN_CSV = DATA_PROCESSED_DIR / "telco_clean.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPECTED_ROWS = 7043
EXPECTED_COLUMNS = 21

ID_COL = "customerID"
TARGET_COL = "Churn"

# Business assumptions (IDR). Edit these to match your own context; Phase 5 reads them.
RETENTION_COST_IDR = 50_000           # cost of one retention offer (discount, call, voucher)
CHURN_LOSS_IDR = 1_500_000            # revenue lost when a customer churns (approx. CLV)
RETENTION_SUCCESS_RATE = 0.5          # share of contacted would-be churners who actually stay
