import os
from typing import Optional

from pydantic_settings import SettingsConfigDict

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.utils.env import CondaConf, Env, LocalEnv


class FactorCoSTEERSettings(CoSTEERSettings):
    model_config = SettingsConfigDict(env_prefix="FACTOR_CoSTEER_")

    data_folder: str = "git_ignore_folder/factor_implementation_source_data"
    """Path to the folder containing financial data (default is fundamental data in Qlib)"""

    data_folder_debug: str = "git_ignore_folder/factor_implementation_source_data_debug"
    """Path to the folder containing partial financial data (for debugging)"""

    simple_background: bool = False
    """Whether to use simple background information for code feedback"""

    file_based_execution_timeout: int = 3600
    """Timeout in seconds for each factor implementation execution"""

    select_method: str = "random"
    """Method for the selection of factors implementation"""

    python_bin: str = "python"
    """Path to the Python binary"""


def get_factor_env(
    conf_type: Optional[str] = None,
    extra_volumes: dict = {},
    running_timeout_period: int = 600,
    enable_cache: Optional[bool] = None,
) -> Env:
    conf = FactorCoSTEERSettings()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "base")
    venv_bin = os.environ.get("VENV_BIN_PATH", "")
    env_conf = CondaConf(conda_env_name=conda_env)
    # Override bin_path with venv if available (for non-Conda environments)
    if venv_bin:
        # VENV_BIN_PATH may point to the python binary; extract parent directory
        from pathlib import Path
        venv_path = Path(venv_bin)
        env_conf.bin_path = str(venv_path.parent) if venv_path.is_file() else venv_bin
    else:
        # Auto-detect venv bin directory from current interpreter
        import sys
        from pathlib import Path
        env_conf.bin_path = str(Path(sys.executable).parent)
    env = LocalEnv(conf=env_conf)
    env.conf.extra_volumes = extra_volumes.copy()
    env.conf.running_timeout_period = running_timeout_period
    if enable_cache is not None:
        env.conf.enable_cache = enable_cache
    env.prepare()
    return env


FACTOR_COSTEER_SETTINGS = FactorCoSTEERSettings()
