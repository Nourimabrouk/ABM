"""
Export Utilities for High-Quality Figure Output

This module provides sophisticated export capabilities for publication-ready
figures with proper format handling, metadata inclusion, and quality control.

Author: Delta Agent - State-of-the-Art Visualization Specialist
Created: 2025-09-15
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as backend_pdf
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

logger = logging.getLogger(__name__)

class FigureExporter:
    """
    High-quality figure export system for academic publications.

    Provides standardized export with proper resolution, format handling,
    metadata inclusion, and quality control for dissertation figures.
    """

    def __init__(self,
                 output_dir: Path = None,
                 base_dpi: int = 300,
                 formats: List[str] = None):
        """
        Initialize figure exporter.

        Args:
            output_dir: Base output directory for figures
            base_dpi: Base DPI for high-quality output
            formats: List of formats to export by default
        """
        self.output_dir = output_dir or Path("outputs/figures")
        self.base_dpi = base_dpi
        self.formats = formats or ['png', 'pdf', 'svg']

        # Create output directories
        self._create_output_structure()

        # Export settings for different formats
        self.format_settings = {
            'png': {'dpi': self.base_dpi, 'transparent': False, 'optimize': True},
            'pdf': {'dpi': self.base_dpi, 'transparent': False, 'bbox_inches': 'tight'},
            'svg': {'transparent': False, 'bbox_inches': 'tight'},
            'eps': {'dpi': self.base_dpi, 'transparent': False, 'bbox_inches': 'tight'},
            'tiff': {'dpi': self.base_dpi, 'transparent': False, 'compression': 'lzw'},
            'jpg': {'dpi': self.base_dpi, 'quality': 95, 'optimize': True}
        }

        # Metadata template
        self.metadata_template = {
            'title': '',
            'description': '',
            'author': 'Delta Agent - ABM-RSiena Research',
            'creation_date': '',
            'software': 'Python/Matplotlib',
            'project': 'ABM-RSiena Integration PhD Research',
            'keywords': ['agent-based modeling', 'social networks', 'RSiena', 'statistical sociology'],
            'figure_type': '',
            'data_sources': [],
            'statistical_methods': [],
            'color_palette': '',
            'accessibility': True
        }

        logger.info(f"Figure exporter initialized, output: {self.output_dir}")

    def _create_output_structure(self):
        """Create organized output directory structure."""
        directories = [
            self.output_dir,
            self.output_dir / "publication_ready",
            self.output_dir / "high_resolution",
            self.output_dir / "web_optimized",
            self.output_dir / "presentation",
            self.output_dir / "supplementary",
            self.output_dir / "raw",
            self.output_dir / "metadata"
        ]

        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    def save_figure(self,
                   figure: Figure,
                   filename: str,
                   format: str = 'png',
                   quality_level: str = 'publication',
                   metadata: Optional[Dict[str, Any]] = None,
                   **kwargs) -> Path:
        """
        Save figure with specified quality and format settings.

        Args:
            figure: Matplotlib figure object
            filename: Base filename (without extension)
            format: Output format ('png', 'pdf', 'svg', etc.)
            quality_level: Quality preset ('publication', 'presentation', 'web', 'print')
            metadata: Additional metadata for the figure
            **kwargs: Additional arguments for savefig

        Returns:
            Path to saved figure
        """
        # Determine output directory based on quality level
        quality_dirs = {
            'publication': self.output_dir / "publication_ready",
            'presentation': self.output_dir / "presentation",
            'web': self.output_dir / "web_optimized",
            'print': self.output_dir / "high_resolution",
            'raw': self.output_dir / "raw"
        }

        output_dir = quality_dirs.get(quality_level, self.output_dir / "publication_ready")

        # Get format-specific settings
        save_kwargs = self.format_settings.get(format, {}).copy()
        save_kwargs.update(kwargs)

        # Adjust settings based on quality level
        if quality_level == 'presentation':
            save_kwargs['dpi'] = min(save_kwargs.get('dpi', 150), 150)
        elif quality_level == 'web':
            save_kwargs['dpi'] = min(save_kwargs.get('dpi', 72), 72)
            save_kwargs['optimize'] = True
        elif quality_level == 'print':
            save_kwargs['dpi'] = max(save_kwargs.get('dpi', 600), 600)

        # Create full filepath
        filepath = output_dir / f"{filename}.{format}"

        # Save figure
        try:
            figure.savefig(filepath, format=format, **save_kwargs)

            # Save metadata
            self._save_metadata(filepath, metadata, figure)

            # Validate output
            self._validate_output(filepath, format)

            logger.info(f"Figure saved: {filepath} ({quality_level} quality)")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save figure {filepath}: {e}")
            raise

    def save_figure_multiple_formats(self,
                                   figure: Figure,
                                   filename: str,
                                   formats: Optional[List[str]] = None,
                                   quality_level: str = 'publication',
                                   metadata: Optional[Dict[str, Any]] = None) -> List[Path]:
        """
        Save figure in multiple formats simultaneously.

        Args:
            figure: Matplotlib figure object
            filename: Base filename
            formats: List of formats to save (None for default)
            quality_level: Quality preset
            metadata: Figure metadata

        Returns:
            List of paths to saved figures
        """
        formats = formats or self.formats
        saved_paths = []

        for format in formats:
            try:
                path = self.save_figure(figure, filename, format, quality_level, metadata)
                saved_paths.append(path)
            except Exception as e:
                logger.warning(f"Failed to save {format} format for {filename}: {e}")

        logger.info(f"Saved figure in {len(saved_paths)} formats: {filename}")
        return saved_paths

    def create_publication_package(self,
                                 figures: List[Tuple[Figure, str, Dict]],
                                 package_name: str = "publication_figures") -> Path:
        """
        Create a complete publication package with all figures and metadata.

        Args:
            figures: List of (figure, filename, metadata) tuples
            package_name: Name for the package directory

        Returns:
            Path to created package directory
        """
        package_dir = self.output_dir / package_name
        package_dir.mkdir(exist_ok=True)

        # Create subdirectories
        subdirs = ['figures', 'metadata', 'thumbnails', 'source_data']
        for subdir in subdirs:
            (package_dir / subdir).mkdir(exist_ok=True)

        saved_figures = []

        for figure, filename, metadata in figures:
            # Save in multiple formats
            figure_paths = self.save_figure_multiple_formats(
                figure, filename, metadata=metadata
            )

            # Copy to package directory
            for path in figure_paths:
                dest_path = package_dir / "figures" / path.name
                shutil.copy2(path, dest_path)
                saved_figures.append(dest_path)

            # Create thumbnail
            self._create_thumbnail(figure, package_dir / "thumbnails" / f"{filename}_thumb.png")

        # Create package index
        self._create_package_index(package_dir, saved_figures)

        # Create README
        self._create_package_readme(package_dir, figures)

        logger.info(f"Publication package created: {package_dir}")
        return package_dir

    def _save_metadata(self, filepath: Path, metadata: Optional[Dict], figure: Figure):
        """Save metadata for the figure."""
        if metadata is None:
            metadata = {}

        # Merge with template
        full_metadata = self.metadata_template.copy()
        full_metadata.update(metadata)
        full_metadata['creation_date'] = datetime.now().isoformat()
        full_metadata['filepath'] = str(filepath)
        full_metadata['filesize'] = filepath.stat().st_size if filepath.exists() else 0
        full_metadata['figure_size'] = figure.get_size_inches().tolist()
        full_metadata['dpi'] = figure.get_dpi()

        # Save metadata file
        metadata_path = self.output_dir / "metadata" / f"{filepath.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)

    def _validate_output(self, filepath: Path, format: str):
        """Validate the output file."""
        if not filepath.exists():
            raise FileNotFoundError(f"Output file not created: {filepath}")

        if filepath.stat().st_size == 0:
            raise ValueError(f"Empty output file: {filepath}")

        # Format-specific validation
        if format in ['png', 'jpg', 'tiff']:
            try:
                with Image.open(filepath) as img:
                    if img.size[0] == 0 or img.size[1] == 0:
                        raise ValueError(f"Invalid image dimensions: {filepath}")
            except Exception as e:
                logger.warning(f"Image validation failed for {filepath}: {e}")

    def _create_thumbnail(self, figure: Figure, output_path: Path, size: Tuple[int, int] = (300, 200)):
        """Create thumbnail of the figure."""
        try:
            # Save temporary high-res version
            temp_path = output_path.parent / f"temp_{output_path.name}"
            figure.savefig(temp_path, dpi=150, bbox_inches='tight', format='png')

            # Create thumbnail
            with Image.open(temp_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.save(output_path, 'PNG', optimize=True)

            # Clean up
            temp_path.unlink()

        except Exception as e:
            logger.warning(f"Failed to create thumbnail {output_path}: {e}")

    def _create_package_index(self, package_dir: Path, figure_paths: List[Path]):
        """Create an index file for the package."""
        index_data = {
            'package_name': package_dir.name,
            'creation_date': datetime.now().isoformat(),
            'total_figures': len(figure_paths),
            'figures': []
        }

        for path in figure_paths:
            figure_info = {
                'filename': path.name,
                'format': path.suffix[1:],
                'size_bytes': path.stat().st_size,
                'relative_path': str(path.relative_to(package_dir))
            }
            index_data['figures'].append(figure_info)

        index_path = package_dir / "index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def _create_package_readme(self, package_dir: Path, figures: List[Tuple]):
        """Create README file for the package."""
        readme_content = f"""# {package_dir.name.replace('_', ' ').title()}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

This package contains publication-ready figures for ABM-RSiena integration research.

### Directory Structure

- `figures/` - High-quality figures in multiple formats
- `thumbnails/` - Preview thumbnails of all figures
- `metadata/` - Detailed metadata for each figure
- `source_data/` - Source data files (if applicable)

### Figures Included

"""

        for i, (figure, filename, metadata) in enumerate(figures, 1):
            title = metadata.get('title', filename) if metadata else filename
            description = metadata.get('description', 'No description') if metadata else 'No description'

            readme_content += f"{i}. **{title}**\n"
            readme_content += f"   - Filename: {filename}\n"
            readme_content += f"   - Description: {description}\n\n"

        readme_content += """
### Usage Instructions

1. For publication: Use files in `figures/` directory
2. For presentations: Resize as needed (original resolution is 300 DPI)
3. For web use: Consider creating web-optimized versions at 72 DPI

### Technical Specifications

- Resolution: 300 DPI (publication quality)
- Formats: PNG, PDF, SVG (as available)
- Color space: RGB (with CMYK-compatible colors)
- Accessibility: Colorblind-friendly palettes used throughout

### Citation

When using these figures, please cite the original research:
[Add citation information here]

### Contact

For questions about these figures or the underlying research:
Delta Agent - ABM-RSiena Visualization Specialist
"""

        readme_path = package_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)

    def optimize_for_web(self, input_path: Path, output_path: Optional[Path] = None,
                        max_width: int = 1200, quality: int = 85) -> Path:
        """
        Optimize figure for web use.

        Args:
            input_path: Path to input figure
            output_path: Path for optimized output (None for auto)
            max_width: Maximum width in pixels
            quality: JPEG quality (1-95)

        Returns:
            Path to optimized figure
        """
        if output_path is None:
            output_path = self.output_dir / "web_optimized" / f"{input_path.stem}_web.jpg"

        try:
            with Image.open(input_path) as img:
                # Calculate new size maintaining aspect ratio
                if img.width > max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)

                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background

                # Save optimized version
                img.save(output_path, 'JPEG', quality=quality, optimize=True, progressive=True)

            logger.info(f"Web-optimized figure saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to optimize figure for web: {e}")
            raise

    def create_figure_gallery(self,
                            figures_dir: Path = None,
                            output_path: Path = None,
                            columns: int = 3) -> Path:
        """
        Create an HTML gallery of all figures.

        Args:
            figures_dir: Directory containing figures (None for default)
            output_path: Output path for gallery HTML
            columns: Number of columns in gallery

        Returns:
            Path to created gallery
        """
        figures_dir = figures_dir or (self.output_dir / "publication_ready")
        output_path = output_path or (self.output_dir / "gallery.html")

        # Find all figure files
        figure_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.svg']:
            figure_files.extend(figures_dir.glob(ext))

        # Create thumbnails if needed
        thumb_dir = self.output_dir / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABM-RSiena Research Figure Gallery</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .figure-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        .figure-card:hover {{
            transform: translateY(-5px);
        }}
        .figure-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 8px;
            cursor: pointer;
        }}
        .figure-title {{
            font-weight: bold;
            margin: 10px 0 5px 0;
            font-size: 16px;
        }}
        .figure-info {{
            color: #666;
            font-size: 12px;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            width: 90%;
            max-width: 1200px;
            max-height: 90%;
            object-fit: contain;
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ABM-RSiena Integration Research</h1>
        <h2>Publication Figure Gallery</h2>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="gallery">
"""

        for fig_path in sorted(figure_files):
            # Create thumbnail path
            thumb_path = thumb_dir / f"{fig_path.stem}_thumb.png"

            # Get relative paths for HTML
            fig_rel = fig_path.relative_to(self.output_dir)
            thumb_rel = thumb_path.relative_to(self.output_dir) if thumb_path.exists() else fig_rel

            # Get file info
            file_size = fig_path.stat().st_size / 1024  # KB

            html_content += f"""
        <div class="figure-card">
            <img src="{thumb_rel}" alt="{fig_path.stem}" class="figure-image" onclick="openModal('{fig_rel}')">
            <div class="figure-title">{fig_path.stem.replace('_', ' ').title()}</div>
            <div class="figure-info">
                Format: {fig_path.suffix[1:].upper()} | Size: {file_size:.1f} KB
            </div>
        </div>
"""

        html_content += """
    </div>

    <!-- Modal for full-size images -->
    <div id="imageModal" class="modal">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function openModal(imageSrc) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = imageSrc;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close modal when clicking outside image
        window.onclick = function(event) {
            if (event.target == document.getElementById('imageModal')) {
                closeModal();
            }
        }

        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
    </script>
</body>
</html>
"""

        # Write HTML file
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"Figure gallery created: {output_path}")
        return output_path

    def get_figure_stats(self) -> Dict[str, Any]:
        """Get statistics about exported figures."""
        stats = {
            'total_figures': 0,
            'formats': {},
            'quality_levels': {},
            'total_size_mb': 0,
            'directories': {}
        }

        # Scan all subdirectories
        for subdir in self.output_dir.iterdir():
            if subdir.is_dir() and subdir.name != 'metadata':
                dir_stats = {'count': 0, 'size_mb': 0}

                for file in subdir.glob('*'):
                    if file.is_file():
                        stats['total_figures'] += 1
                        dir_stats['count'] += 1

                        file_size_mb = file.stat().st_size / (1024 * 1024)
                        stats['total_size_mb'] += file_size_mb
                        dir_stats['size_mb'] += file_size_mb

                        # Track formats
                        format = file.suffix[1:].lower()
                        stats['formats'][format] = stats['formats'].get(format, 0) + 1

                stats['directories'][subdir.name] = dir_stats

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize exporter
    exporter = FigureExporter()

    # Create sample figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-', linewidth=2, markersize=8)
    ax.set_title('Sample Figure for Export Testing')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid(True, alpha=0.3)

    # Test single format export
    metadata = {
        'title': 'Sample Test Figure',
        'description': 'Testing the export functionality',
        'figure_type': 'line_plot'
    }

    saved_path = exporter.save_figure(fig, 'test_figure', 'png', 'publication', metadata)
    print(f"Figure saved to: {saved_path}")

    # Test multiple format export
    all_paths = exporter.save_figure_multiple_formats(fig, 'test_figure_multi', metadata=metadata)
    print(f"Figure saved in {len(all_paths)} formats")

    # Create gallery
    gallery_path = exporter.create_figure_gallery()
    print(f"Gallery created: {gallery_path}")

    # Get statistics
    stats = exporter.get_figure_stats()
    print(f"Export statistics: {stats}")

    plt.close(fig)