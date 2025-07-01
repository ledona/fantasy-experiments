import warnings

from fantasy_py import log

# TODO: remove warning suppression when it is no longer needed 2025.7.1
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning
)


log.set_default_log_level(only_fantasy=False)
