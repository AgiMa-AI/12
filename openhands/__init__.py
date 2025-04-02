import os

__package_name__ = 'openhands_ai'


def get_version():
    # Try getting the version from pyproject.toml
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root_dir, 'pyproject.toml'), 'r') as f:
            for line in f:
                if line.startswith('version ='):
                    return line.split('=')[1].strip().strip('"')
    except FileNotFoundError:
        pass

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(__package_name__)
    except (ImportError, PackageNotFoundError):
        pass

    try:
        from pkg_resources import DistributionNotFound, get_distribution

        return get_distribution(__package_name__).version
    except (ImportError, DistributionNotFound):
        pass

    return 'unknown'


try:
    __version__ = get_version()
except Exception:
    __version__ = 'unknown'

# Import LangChain Router integration
try:
    from openhands.langchain_router import LangChainRouter, RouterConfig
    from openhands.langchain_router.integration import OpenHandsLangChainIntegration
    
    # Initialize the LangChain Router integration
    langchain_router_integration = None
    
    def get_langchain_router():
        global langchain_router_integration
        if langchain_router_integration is None:
            langchain_router_integration = OpenHandsLangChainIntegration()
        return langchain_router_integration
    
except ImportError:
    # LangChain Router is optional
    def get_langchain_router():
        raise ImportError("LangChain Router is not available. Install the required dependencies.")
