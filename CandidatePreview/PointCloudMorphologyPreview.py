"""
Point Cloud Morphology Preview Module

This module provides preview functions for analyzing point cloud morphology
during candidate preview. It complements the main visualization tools with
focused analysis on individual events.
"""

import inspect
try:
    from eve_smlm.Utils import utilsHelper
    from eve_smlm.EventDistributions import eventDistributions
except ImportError:
    from Utils import utilsHelper
    from EventDistributions import eventDistributions

import pandas as pd
import numpy as np
import logging
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# Required function __function_metadata__
def __function_metadata__():
    return {
        "MorphologyMetricsTable": {
            "required_kwargs": [],
            "optional_kwargs": [
                {"name": "show_advanced", "display_text": "Show advanced metrics",
                 "description": "Display additional morphology metrics", "default": "False"},
            ],
            "help_string": "Displays a comprehensive table of morphology metrics for the point cloud.",
            "display_name": "Morphology Metrics Table"
        },
        "ShapeCharacterization": {
            "required_kwargs": [
                {"name": "view", "display_text": "View plane",
                 "description": "Plane for shape characterization",
                 "default": "xy", "options": ["xy", "xz", "yz"]},
            ],
            "optional_kwargs": [
                {"name": "show_ellipse", "display_text": "Show fitted ellipse",
                 "description": "Fit and display confidence ellipse", "default": "True"},
                {"name": "confidence_level", "display_text": "Confidence level",
                 "description": "Confidence level for ellipse (0-1)", "default": "0.95"},
            ],
            "help_string": "2D shape characterization with fitted ellipses and orientation analysis.",
            "display_name": "2D Shape Characterization"
        },
        "SpatialDistributionAnalysis": {
            "required_kwargs": [],
            "optional_kwargs": [
                {"name": "n_bins", "display_text": "Number of bins",
                 "description": "Bins for distribution histograms", "default": "30"},
            ],
            "help_string": "Analyzes spatial distribution patterns with nearest neighbor and radial distribution analysis.",
            "display_name": "Spatial Distribution Analysis"
        }
    }


#-------------------------------------------------------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------------------------------------------------------

def compute_all_metrics(events_df):
    """Compute comprehensive morphology metrics."""
    metrics = {}
    
    coords = events_df[['x', 'y', 't']].values
    n_events = len(coords)
    
    if n_events < 4:
        return metrics
    
    # Basic stats
    centroid = np.mean(coords, axis=0)
    metrics['N events'] = n_events
    metrics['Centroid X (px)'] = f"{centroid[0]:.2f}"
    metrics['Centroid Y (px)'] = f"{centroid[1]:.2f}"
    metrics['Centroid T (μs)'] = f"{centroid[2]:.2f}"
    
    # Ranges
    ranges = np.ptp(coords, axis=0)
    metrics['X Range (px)'] = f"{ranges[0]:.2f}"
    metrics['Y Range (px)'] = f"{ranges[1]:.2f}"
    metrics['T Range (μs)'] = f"{ranges[2]:.2f}"
    
    # Volume and density
    volume = np.prod(ranges)
    metrics['Volume (px²·μs)'] = f"{volume:.2f}"
    metrics['Density (events/vol)'] = f"{n_events/volume:.4f}" if volume > 0 else "N/A"
    
    # Convex hull
    try:
        hull = ConvexHull(coords)
        metrics['Hull Volume'] = f"{hull.volume:.2f}"
        metrics['Hull Area'] = f"{hull.area:.2f}"
        metrics['Compactness'] = f"{n_events/hull.volume:.4f}" if hull.volume > 0 else "N/A"
    except:
        metrics['Hull Volume'] = "N/A"
        metrics['Hull Area'] = "N/A"
        metrics['Compactness'] = "N/A"
    
    # PCA
    try:
        centered = coords - centroid
        pca = PCA(n_components=3)
        pca.fit(centered)
        
        ev = pca.explained_variance_
        metrics['PC1 Variance'] = f"{ev[0]:.2f}"
        metrics['PC2 Variance'] = f"{ev[1]:.2f}"
        metrics['PC3 Variance'] = f"{ev[2]:.2f}"
        metrics['PC1 Ratio'] = f"{pca.explained_variance_ratio_[0]:.3f}"
        metrics['PC2 Ratio'] = f"{pca.explained_variance_ratio_[1]:.3f}"
        metrics['PC3 Ratio'] = f"{pca.explained_variance_ratio_[2]:.3f}"
        
        # Shape descriptors
        metrics['Anisotropy'] = f"{(ev[0]-ev[2])/ev[0]:.3f}" if ev[0] > 0 else "N/A"
        metrics['Linearity'] = f"{(ev[0]-ev[1])/ev[0]:.3f}" if ev[0] > 0 else "N/A"
        metrics['Planarity'] = f"{(ev[1]-ev[2])/ev[0]:.3f}" if ev[0] > 0 else "N/A"
        metrics['Sphericity'] = f"{ev[2]/ev[0]:.3f}" if ev[0] > 0 else "N/A"
    except:
        pass
    
    # Nearest neighbors
    if n_events > 1:
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)
        nn_distances = np.min(dist_mat, axis=1)
        
        metrics['Mean NN Dist'] = f"{np.mean(nn_distances):.2f}"
        metrics['Std NN Dist'] = f"{np.std(nn_distances):.2f}"
        metrics['Min NN Dist'] = f"{np.min(nn_distances):.2f}"
        metrics['Max NN Dist'] = f"{np.max(nn_distances):.2f}"
    
    return metrics


def fit_ellipse_2d(points, confidence=0.95):
    """
    Fit confidence ellipse to 2D points.
    
    Returns ellipse parameters: (center_x, center_y, width, height, angle)
    """
    from scipy.stats import chi2
    
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    cov = np.cov(points_centered.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Calculate angle
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Calculate ellipse dimensions based on confidence level
    chi2_val = chi2.ppf(confidence, df=2)
    width, height = 2 * np.sqrt(eigenvalues * chi2_val)
    
    return center[0], center[1], width, height, angle


#-------------------------------------------------------------------------------------------------------------------------------
# Callable functions
#-------------------------------------------------------------------------------------------------------------------------------

def MorphologyMetricsTable(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """Display comprehensive morphology metrics in a table format."""
    
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    show_advanced = utilsHelper.strtobool(kwargs.get('show_advanced', 'False'))
    
    events = findingResult.copy()
    
    if len(events) < 1:
        ax = figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No events to analyze', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return 1
    
    # Compute metrics
    metrics = compute_all_metrics(events)
    
    if not show_advanced:
        # Filter to basic metrics
        basic_keys = ['N events', 'Centroid X (px)', 'Centroid Y (px)', 'Centroid T (μs)',
                     'X Range (px)', 'Y Range (px)', 'T Range (μs)',
                     'Volume (px²·μs)', 'Density (events/vol)',
                     'Anisotropy', 'Linearity', 'Planarity', 'Mean NN Dist']
        metrics = {k: v for k, v in metrics.items() if k in basic_keys}
    
    # Create table visualization
    ax = figure.add_subplot(111)
    ax.axis('off')
    
    # Prepare table data
    table_data = [[k, v] for k, v in metrics.items()]
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        if i == 0:
            # Header row
            for j in range(2):
                cell = table[(i, j)]
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
        else:
            # Alternating row colors
            for j in range(2):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
    
    ax.set_title('Point Cloud Morphology Metrics', fontsize=12, fontweight='bold', pad=20)
    
    figure.tight_layout()
    
    return 1


def ShapeCharacterization(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """2D shape characterization with fitted ellipses and orientation."""
    
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    pixel_size = float(settings['PixelSize_nm']['value'])
    
    view = kwargs.get('view', 'xy')
    show_ellipse = utilsHelper.strtobool(kwargs.get('show_ellipse', 'True'))
    confidence_level = float(kwargs.get('confidence_level', 0.95))
    
    events = findingResult.copy()
    
    if len(events) < 3:
        logging.warning("Too few events for shape characterization")
        return 1
    
    # Extract coordinates based on view
    if view == 'xy':
        coords = events[['x', 'y']].values
        xlabel, ylabel = 'x [px]', 'y [px]'
    elif view == 'xz':
        coords = events[['x', 't']].values
        coords[:, 1] *= 1e-3  # Convert to ms
        xlabel, ylabel = 'x [px]', 't [ms]'
    elif view == 'yz':
        coords = events[['y', 't']].values
        coords[:, 1] *= 1e-3
        xlabel, ylabel = 'y [px]', 't [ms]'
    else:
        logging.error(f"Unknown view: {view}")
        return 1
    
    # Create plot
    ax = figure.add_subplot(111)
    
    # Scatter events
    ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.6, c='C0', 
              edgecolors='black', linewidth=0.5, label='Events')
    
    # Fit and plot ellipse
    if show_ellipse and len(coords) >= 5:
        try:
            cx, cy, width, height, angle = fit_ellipse_2d(coords, confidence_level)
            
            ellipse = Ellipse((cx, cy), width, height, angle=angle,
                            facecolor='none', edgecolor='red', linewidth=2,
                            linestyle='--', label=f'{confidence_level*100:.0f}% Confidence Ellipse')
            ax.add_patch(ellipse)
            
            # Plot center
            ax.plot(cx, cy, 'rx', markersize=12, markeredgewidth=2, label='Centroid')
            
            # Plot principal axes
            # Major axis
            dx_major = width/2 * np.cos(np.radians(angle))
            dy_major = width/2 * np.sin(np.radians(angle))
            ax.plot([cx - dx_major, cx + dx_major],
                   [cy - dy_major, cy + dy_major],
                   'r-', linewidth=2, alpha=0.7, label='Major Axis')
            
            # Minor axis
            dx_minor = height/2 * np.cos(np.radians(angle + 90))
            dy_minor = height/2 * np.sin(np.radians(angle + 90))
            ax.plot([cx - dx_minor, cx + dx_minor],
                   [cy - dy_minor, cy + dy_minor],
                   'b-', linewidth=2, alpha=0.7, label='Minor Axis')
            
            # Add metrics text
            eccentricity = np.sqrt(1 - (min(width, height)**2 / max(width, height)**2))
            aspect_ratio = max(width, height) / min(width, height)
            
            metrics_text = f"Angle: {angle:.1f}°\n"
            metrics_text += f"Eccentricity: {eccentricity:.3f}\n"
            metrics_text += f"Aspect Ratio: {aspect_ratio:.2f}"
            
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
        except Exception as e:
            logging.warning(f"Could not fit ellipse: {e}")
    
    # Plot localizations
    if len(fittingResult) > 0:
        if view == 'xy':
            ax.plot(fittingResult['x']/pixel_size, fittingResult['y']/pixel_size,
                   'g*', markersize=15, markeredgewidth=1.5, 
                   markeredgecolor='black', label='Localization')
        elif view == 'xz':
            ax.plot(fittingResult['x']/pixel_size, fittingResult['t'],
                   'g*', markersize=15, markeredgewidth=1.5,
                   markeredgecolor='black', label='Localization')
        elif view == 'yz':
            ax.plot(fittingResult['y']/pixel_size, fittingResult['t'],
                   'g*', markersize=15, markeredgewidth=1.5,
                   markeredgecolor='black', label='Localization')
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'Shape Characterization - {view.upper()} Plane', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='datalim')
    
    figure.tight_layout()
    
    return 1


def SpatialDistributionAnalysis(findingResult, fittingResult, previewEvents, figure, settings, **kwargs):
    """Analyze spatial distribution patterns."""
    
    [provided_optional_args, missing_optional_args] = utilsHelper.argumentChecking(
        __function_metadata__(), inspect.currentframe().f_code.co_name, kwargs)
    
    n_bins = int(kwargs.get('n_bins', 30))
    
    events = findingResult.copy()
    
    if len(events) < 2:
        logging.warning("Too few events for distribution analysis")
        return 1
    
    coords = events[['x', 'y', 't']].values
    
    # Create subplot layout
    fig = figure
    ax1 = fig.add_subplot(221)  # NN distance histogram
    ax2 = fig.add_subplot(222)  # Radial distribution
    ax3 = fig.add_subplot(223)  # X,Y,T distributions
    ax4 = fig.add_subplot(224)  # Distance matrix heatmap
    
    # 1. Nearest neighbor distance distribution
    if len(coords) > 1:
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)
        nn_distances = np.min(dist_mat, axis=1)
        
        ax1.hist(nn_distances, bins=n_bins, alpha=0.7, color='C0', edgecolor='black')
        ax1.axvline(np.mean(nn_distances), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(nn_distances):.2f}')
        ax1.set_xlabel('Nearest Neighbor Distance', fontsize=9)
        ax1.set_ylabel('Frequency', fontsize=9)
        ax1.set_title('NN Distance Distribution', fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
    
    # 2. Radial distribution from centroid
    centroid = np.mean(coords, axis=0)
    radial_distances = np.linalg.norm(coords - centroid, axis=1)
    
    ax2.hist(radial_distances, bins=n_bins, alpha=0.7, color='C1', edgecolor='black')
    ax2.axvline(np.mean(radial_distances), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(radial_distances):.2f}')
    ax2.set_xlabel('Radial Distance from Centroid', fontsize=9)
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.set_title('Radial Distribution', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Coordinate distributions
    ax3.hist(coords[:, 0], bins=n_bins//2, alpha=0.5, color='C0', 
            label='X', edgecolor='black')
    ax3.hist(coords[:, 1], bins=n_bins//2, alpha=0.5, color='C1',
            label='Y', edgecolor='black')
    ax3.hist(coords[:, 2], bins=n_bins//2, alpha=0.5, color='C2',
            label='T', edgecolor='black')
    ax3.set_xlabel('Coordinate Value', fontsize=9)
    ax3.set_ylabel('Frequency', fontsize=9)
    ax3.set_title('Coordinate Distributions', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Distance matrix heatmap (sample if too large)
    max_sample = 100
    if len(coords) > max_sample:
        sample_idx = np.random.choice(len(coords), max_sample, replace=False)
        sample_coords = coords[sample_idx]
        sample_dist_mat = distance_matrix(sample_coords, sample_coords)
        title_suffix = f' (sampled {max_sample})'
    else:
        sample_dist_mat = distance_matrix(coords, coords)
        title_suffix = ''
    
    im = ax4.imshow(sample_dist_mat, cmap='hot', aspect='auto', 
                    interpolation='nearest')
    ax4.set_xlabel('Event Index', fontsize=9)
    ax4.set_ylabel('Event Index', fontsize=9)
    ax4.set_title(f'Distance Matrix{title_suffix}', fontsize=10, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax4)
    cbar.set_label('Distance', fontsize=8)
    
    fig.tight_layout()
    
    return 1
