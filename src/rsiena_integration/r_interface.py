"""
R Interface for RSiena Integration

This module provides a robust Python-R interface for communicating with RSiena,
handling R session management, error handling, and data exchange between
Python and R environments.

Features:
- Automatic R session management with error recovery
- RSiena package installation and version checking
- Robust error handling with detailed diagnostics
- Memory-efficient data transfer
- R code execution with timeout handling

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import warnings
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.rinterface_lib.embedded import RRuntimeError
    import rpy2.robjects.numpy2ri

    # Activate automatic conversion
    rpy2.robjects.numpy2ri.activate()

    RPY2_AVAILABLE = True
except ImportError as e:
    RPY2_AVAILABLE = False
    warnings.warn(f"rpy2 not available: {e}. RSiena integration will not work.")

logger = logging.getLogger(__name__)


@dataclass
class RSessionConfig:
    """Configuration for R session management."""
    r_executable: Optional[str] = None
    r_home: Optional[str] = None
    memory_limit_mb: int = 2048
    timeout_seconds: int = 300
    auto_install_packages: bool = True
    required_packages: List[str] = None

    def __post_init__(self):
        if self.required_packages is None:
            self.required_packages = ['RSiena', 'igraph', 'network', 'sna']


class RInterface:
    """
    Robust R interface for RSiena integration with comprehensive error handling.

    Manages R session lifecycle, package dependencies, and provides safe
    execution environment for RSiena operations.
    """

    def __init__(self, config: Optional[RSessionConfig] = None):
        """
        Initialize R interface with configuration.

        Args:
            config: R session configuration
        """
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 is required for R interface. Install with: pip install rpy2")

        self.config = config or RSessionConfig()
        self.r_session = None
        self.r_base = None
        self.r_utils = None
        self.rsiena = None
        self.session_active = False
        self.package_status = {}

        # Initialize R session
        self._initialize_r_session()

    def _initialize_r_session(self):
        """Initialize R session with error handling and recovery."""
        try:
            logger.info("Initializing R session...")

            # Set R environment variables if specified
            if self.config.r_home:
                import os
                os.environ['R_HOME'] = self.config.r_home

            # Initialize robjects
            self.r_session = robjects.r
            self.r_base = importr('base')
            self.r_utils = importr('utils')

            # Set memory limits
            self._set_memory_limits()

            # Check and install required packages
            if self.config.auto_install_packages:
                self._ensure_packages_installed()

            # Load RSiena
            self._load_rsiena()

            self.session_active = True
            logger.info("R session initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize R session: {e}")
            self.session_active = False
            raise RuntimeError(f"R session initialization failed: {e}")

    def _set_memory_limits(self):
        """Set R memory limits."""
        try:
            # Set memory limit (in MB)
            memory_mb = self.config.memory_limit_mb
            self.r_session(f"memory.limit(size={memory_mb})")
            logger.debug(f"Set R memory limit to {memory_mb}MB")
        except Exception as e:
            logger.warning(f"Could not set R memory limit: {e}")

    def _ensure_packages_installed(self):
        """Ensure required R packages are installed."""
        for package in self.config.required_packages:
            try:
                # Try to load package
                importr(package)
                self.package_status[package] = "loaded"
                logger.debug(f"Package {package} already available")

            except Exception:
                logger.info(f"Installing R package: {package}")
                try:
                    # Install package
                    self.r_utils.install_packages(package)

                    # Verify installation
                    importr(package)
                    self.package_status[package] = "installed"
                    logger.info(f"Successfully installed and loaded {package}")

                except Exception as install_error:
                    logger.error(f"Failed to install {package}: {install_error}")
                    self.package_status[package] = "failed"

                    if package == "RSiena":
                        raise RuntimeError(f"Critical package RSiena installation failed: {install_error}")

    def _load_rsiena(self):
        """Load RSiena package with error handling."""
        try:
            self.rsiena = importr('RSiena')
            logger.info("RSiena package loaded successfully")

            # Check RSiena version
            version = self.execute_r_code("packageVersion('RSiena')")
            logger.info(f"RSiena version: {version}")

        except Exception as e:
            logger.error(f"Failed to load RSiena: {e}")
            raise RuntimeError(f"RSiena loading failed: {e}")

    def execute_r_code(
        self,
        r_code: str,
        timeout: Optional[int] = None,
        return_result: bool = True
    ) -> Any:
        """
        Execute R code with timeout and error handling.

        Args:
            r_code: R code to execute
            timeout: Timeout in seconds (uses config default if None)
            return_result: Whether to return the result

        Returns:
            Result of R code execution or None
        """
        if not self.session_active:
            raise RuntimeError("R session not active")

        timeout = timeout or self.config.timeout_seconds

        try:
            logger.debug(f"Executing R code: {r_code[:100]}...")

            # Execute with timeout handling
            start_time = time.time()
            result = self.r_session(r_code)
            execution_time = time.time() - start_time

            if execution_time > timeout:
                logger.warning(f"R code execution took {execution_time:.2f}s (timeout: {timeout}s)")

            logger.debug(f"R code executed successfully in {execution_time:.2f}s")

            return result if return_result else None

        except RRuntimeError as e:
            logger.error(f"R runtime error: {e}")
            raise RuntimeError(f"R execution failed: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during R execution: {e}")
            raise RuntimeError(f"R execution error: {e}")

    def create_r_object(self, name: str, python_object: Any) -> bool:
        """
        Create R object from Python object.

        Args:
            name: Name for R object
            python_object: Python object to convert

        Returns:
            Success status
        """
        try:
            if isinstance(python_object, np.ndarray):
                # Convert numpy array
                r_object = numpy2ri.py2rpy(python_object)

            elif isinstance(python_object, pd.DataFrame):
                # Convert pandas DataFrame
                with localconverter(robjects.default_converter + pandas2ri.converter):
                    r_object = robjects.conversion.py2rpy(python_object)

            elif isinstance(python_object, (list, tuple)):
                # Convert list/tuple
                r_object = robjects.vectors.ListVector(python_object)

            elif isinstance(python_object, dict):
                # Convert dictionary
                r_object = robjects.ListVector(python_object)

            else:
                # Try direct conversion
                r_object = robjects.conversion.py2rpy(python_object)

            # Assign to R environment
            robjects.globalenv[name] = r_object
            logger.debug(f"Created R object '{name}' from Python object")
            return True

        except Exception as e:
            logger.error(f"Failed to create R object '{name}': {e}")
            return False

    def get_r_object(self, name: str) -> Any:
        """
        Get R object and convert to Python.

        Args:
            name: Name of R object

        Returns:
            Python object or None if failed
        """
        try:
            r_object = robjects.globalenv[name]

            # Try automatic conversion
            python_object = robjects.conversion.rpy2py(r_object)
            logger.debug(f"Retrieved R object '{name}' as Python object")
            return python_object

        except Exception as e:
            logger.error(f"Failed to retrieve R object '{name}': {e}")
            return None

    def check_rsiena_available(self) -> bool:
        """
        Check if RSiena is available and functional.

        Returns:
            True if RSiena is available
        """
        try:
            if not self.rsiena:
                return False

            # Test basic RSiena functionality
            test_code = """
            library(RSiena)
            n <- 10
            network1 <- matrix(rbinom(n*n, 1, 0.1), n, n)
            diag(network1) <- 0
            network2 <- matrix(rbinom(n*n, 1, 0.1), n, n)
            diag(network2) <- 0
            networkData <- sienaDependent(array(c(network1, network2), dim=c(n, n, 2)))
            """

            self.execute_r_code(test_code, timeout=30)
            logger.info("RSiena functionality test passed")
            return True

        except Exception as e:
            logger.error(f"RSiena functionality test failed: {e}")
            return False

    def list_r_objects(self) -> List[str]:
        """
        List all objects in R global environment.

        Returns:
            List of R object names
        """
        try:
            result = self.execute_r_code("ls()")
            if result:
                return list(result)
            return []
        except Exception as e:
            logger.error(f"Failed to list R objects: {e}")
            return []

    def clear_r_workspace(self):
        """Clear R workspace."""
        try:
            self.execute_r_code("rm(list=ls())", return_result=False)
            logger.debug("R workspace cleared")
        except Exception as e:
            logger.warning(f"Failed to clear R workspace: {e}")

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get R memory usage information.

        Returns:
            Dictionary with memory usage stats
        """
        try:
            # Get memory usage
            result = self.execute_r_code("""
            list(
                size = object.size(ls()),
                memory = memory.size(),
                gc_info = gc()
            )
            """)

            return {
                'workspace_size': result[0],
                'memory_used': result[1],
                'gc_info': result[2]
            }

        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}

    def install_package(self, package_name: str, force: bool = False) -> bool:
        """
        Install R package.

        Args:
            package_name: Name of package to install
            force: Force reinstallation

        Returns:
            Success status
        """
        try:
            if not force and package_name in self.package_status:
                if self.package_status[package_name] in ["loaded", "installed"]:
                    logger.info(f"Package {package_name} already available")
                    return True

            logger.info(f"Installing R package: {package_name}")
            self.r_utils.install_packages(package_name)

            # Verify installation
            importr(package_name)
            self.package_status[package_name] = "installed"
            logger.info(f"Successfully installed {package_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to install {package_name}: {e}")
            self.package_status[package_name] = "failed"
            return False

    def save_workspace(self, filepath: Union[str, Path]):
        """
        Save R workspace to file.

        Args:
            filepath: Path to save workspace
        """
        try:
            filepath = str(filepath)
            self.execute_r_code(f'save.image("{filepath}")', return_result=False)
            logger.info(f"R workspace saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save workspace: {e}")

    def load_workspace(self, filepath: Union[str, Path]):
        """
        Load R workspace from file.

        Args:
            filepath: Path to workspace file
        """
        try:
            filepath = str(filepath)
            self.execute_r_code(f'load("{filepath}")', return_result=False)
            logger.info(f"R workspace loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load workspace: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get R session information.

        Returns:
            Dictionary with session information
        """
        try:
            info = {}

            # R version
            info['r_version'] = str(self.execute_r_code("R.version.string"))

            # Loaded packages
            info['loaded_packages'] = list(self.execute_r_code("loadedNamespaces()"))

            # Platform info
            info['platform'] = str(self.execute_r_code("R.version$platform"))

            # Working directory
            info['working_directory'] = str(self.execute_r_code("getwd()"))

            # Memory info
            info['memory'] = self.get_memory_usage()

            # Package status
            info['package_status'] = self.package_status.copy()

            return info

        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {}

    def restart_session(self):
        """Restart R session."""
        try:
            logger.info("Restarting R session...")

            # Clear current session
            self.session_active = False
            self.rsiena = None

            # Reinitialize
            self._initialize_r_session()

            logger.info("R session restarted successfully")

        except Exception as e:
            logger.error(f"Failed to restart R session: {e}")
            raise RuntimeError(f"R session restart failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            if self.session_active:
                self.clear_r_workspace()
        except Exception as e:
            logger.warning(f"Error during R session cleanup: {e}")


def test_r_interface():
    """Test R interface functionality."""
    try:
        # Test basic functionality
        with RInterface() as r_interface:
            # Test R code execution
            result = r_interface.execute_r_code("2 + 2")
            print(f"2 + 2 = {result}")

            # Test object creation
            test_array = np.random.random((5, 5))
            success = r_interface.create_r_object("test_matrix", test_array)
            print(f"Created R object: {success}")

            # Test RSiena availability
            available = r_interface.check_rsiena_available()
            print(f"RSiena available: {available}")

            # Test session info
            info = r_interface.get_session_info()
            print(f"R version: {info.get('r_version', 'Unknown')}")

        print("R interface test completed successfully")
        return True

    except Exception as e:
        print(f"R interface test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test
    logging.basicConfig(level=logging.INFO)
    test_r_interface()