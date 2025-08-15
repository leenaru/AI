from pydantic_settings import BaseSettings
from pydantic import Field
import yaml, os

class Settings(BaseSettings):
    app_config: str = Field(default=os.getenv("APP_CONFIG", "configs/app.yaml"))
    cfg: dict = {}

    def load(self):
        path = self.app_config
        # fallback to example if not present
        if not os.path.exists(path):
            path = "configs/app.yaml.example"
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        return self

settings = Settings().load()
