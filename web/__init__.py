from .web_browser import WebBrowser
from .search import WebSearchComponent  # Use this one as the default
from .selenium import WebCSEComponent  # Updated to new component name

__all__ = ['WebBrowser', 'WebSearchComponent', 'WebCSEComponent']  # Updated in exports list