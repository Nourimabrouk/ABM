"""
R Environment Setup Script for RSiena Integration

This script helps set up R and RSiena for use with the ABM project.
Run this script after installing R on your system.

Installation steps:
1. Download and install R from: https://cran.r-project.org/bin/windows/base/
2. Optionally install RStudio: https://posit.co/download/rstudio-desktop/
3. Run this script to install required packages
"""

import subprocess
import sys
import os
from pathlib import Path


def check_r_installation():
    """Check if R is installed and accessible."""
    try:
        result = subprocess.run(['R', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ R is installed: {result.stdout.split()[2]}")
            return True
        else:
            print("✗ R is not properly installed or not in PATH")
            return False
    except FileNotFoundError:
        print("✗ R is not installed or not in PATH")
        return False


def install_r_packages():
    """Install required R packages for RSiena integration."""
    packages = [
        'RSiena',
        'network',
        'sna',
        'igraph',
        'lattice'
    ]

    # Create R script for package installation
    r_script = f"""
# Install required packages for RSiena integration
packages <- c({', '.join([f'"{pkg}"' for pkg in packages])})

# Function to install packages if not already installed
install_if_missing <- function(pkg) {{
    if (!require(pkg, character.only = TRUE)) {{
        cat("Installing", pkg, "...\\n")
        install.packages(pkg, repos = "https://cran.rstudio.com/")
        library(pkg, character.only = TRUE)
    }} else {{
        cat(pkg, "is already installed\\n")
    }}
}}

# Install packages
for (pkg in packages) {{
    install_if_missing(pkg)
}}

# Test RSiena installation
cat("\\nTesting RSiena installation...\\n")
library(RSiena)
cat("RSiena version:", packageVersion("RSiena"), "\\n")
cat("✓ RSiena is working correctly!\\n")
"""

    # Write R script to temporary file
    script_path = Path("temp_install_packages.R")
    with open(script_path, 'w') as f:
        f.write(r_script)

    try:
        print("Installing R packages...")
        result = subprocess.run(['R', '--vanilla', '-f', str(script_path)],
                              capture_output=True, text=True)

        print("Installation output:")
        print(result.stdout)

        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)

        if result.returncode == 0:
            print("✓ R packages installed successfully!")
        else:
            print("✗ Package installation failed")
            return False

    except Exception as e:
        print(f"Error running R script: {e}")
        return False
    finally:
        # Clean up temporary script
        if script_path.exists():
            script_path.unlink()

    return True


def test_rpy2_integration():
    """Test rpy2 integration with R."""
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr

        # Test basic R connection
        r = robjects.r
        result = r('R.version.string')
        print(f"✓ rpy2 can connect to R: {result[0]}")

        # Test RSiena import
        try:
            rsiena = importr('RSiena')
            print("✓ RSiena can be imported through rpy2")
            return True
        except Exception as e:
            print(f"✗ Cannot import RSiena through rpy2: {e}")
            return False

    except ImportError as e:
        print(f"✗ rpy2 not available: {e}")
        print("Install with: pip install rpy2")
        return False


def main():
    """Main setup function."""
    print("=== R Environment Setup for ABM Project ===\n")

    # Check R installation
    if not check_r_installation():
        print("\nPlease install R first:")
        print("1. Go to: https://cran.r-project.org/bin/windows/base/")
        print("2. Download and install R for Windows")
        print("3. Make sure R is added to your PATH")
        print("4. Restart your terminal/IDE")
        print("5. Run this script again")
        return False

    # Install R packages
    if not install_r_packages():
        return False

    # Test rpy2 integration
    print("\nTesting rpy2 integration...")
    if not test_rpy2_integration():
        print("\nTroubleshooting rpy2:")
        print("1. Make sure rpy2 is installed: pip install rpy2")
        print("2. Check R_HOME environment variable")
        print("3. Restart your Python environment")
        return False

    print("\n✓ R environment setup complete!")
    print("\nNext steps:")
    print("1. Try running the RSiena integration test")
    print("2. Create your first ABM model with RSiena validation")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)