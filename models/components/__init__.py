"""Models Components Module"""

from .base import BaseFineTuner, BaseModelApp, BaseModelRepository, BaseCloudAPI
from .factory import FineTunerFactory, ModelAppFactory, ModelRepositoryFactory, CloudAPIFactory

# Manual component registration for essential components
def register_essential_components():
    """Register essential components for demos."""
    try:
        # Register Ollama model app
        from .model_apps.ollama.ollama_app import OllamaApp
        ModelAppFactory.register('ollama', OllamaApp)
    except ImportError:
        pass
    
    try:
        # Register PyTorch fine-tuner
        from .fine_tuners.pytorch.pytorch_fine_tuner import PyTorchFineTuner
        FineTunerFactory.register('pytorch', PyTorchFineTuner)
    except ImportError:
        pass
    
    try:
        # Register LlamaFactory fine-tuner
        from .fine_tuners.llamafactory.llamafactory_fine_tuner import LlamaFactoryFineTuner
        FineTunerFactory.register('llamafactory', LlamaFactoryFineTuner)
    except ImportError:
        pass

# Register components on import
register_essential_components()

__all__ = [
    "BaseFineTuner",
    "BaseModelApp", 
    "BaseModelRepository",
    "BaseCloudAPI",
    "FineTunerFactory",
    "ModelAppFactory",
    "ModelRepositoryFactory",
    "CloudAPIFactory"
]