from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WebConfig:
    secret_key: str = "hierfl-dev-secret-key"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 5000
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def uploads_dir(self) -> Path:
        return self.project_root / "uploads"

    @property
    def recognized_result_path(self) -> Path:
        return self.results_dir / "recognized_result.json"


WEB_CONFIG = WebConfig()
