import warnings

# TODO: remove warning suppression when it is no longer needed 2025.7.1
warnings.filterwarnings(
    "ignore", message=".*pkg_resources is deprecated as an API.*", category=UserWarning
)
