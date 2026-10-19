from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.artifacts / "figures"

    @property
    def database(self) -> Path:
        return self.processed / "claims_risk.db"


PATHS = ProjectPaths(Path(__file__).resolve().parents[2])
RANDOM_SEED = 42

DATASET_RECORD = "18138102"
DATA_FILES = {
    "Train-1542865627584.csv": "d8784fd271fdb452fe752256ca7e9084",
    "Train_Beneficiarydata-1542865627584.csv": "3d07fc96bcca695b662086aa2e0e3e5c",
    "Train_Inpatientdata-1542865627584.csv": "312e403f49c07a081c4422896e7e66d2",
    "Train_Outpatientdata-1542865627584.csv": "deda74404e2d5b880f305218bb50c1e5",
}
