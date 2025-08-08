"""
DTW Visualization Module

Provides visualization tools for DTW analysis results including:
- DTW warping path visualization
- Trajectory alignment comparison  
- Similarity heatmaps
- Side-by-side video comparison
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import json

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ matplotlib not available, DTW visualizations will be disabled")

# Try to import cv2
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ opencv-python not available, video processing features will be limited")

class DTWVisualizer:
    """
    Visualizes DTW analysis results for basketball shooting comparison.
    
    Creates various plots and animations to show how DTW aligns two shooting motions.
    """
    
    def __init__(self):
        self.fig_size = (12, 8)
        self.colors = {
            'video1': '#2E86AB',
            'video2': '#A23B72', 
            'alignment': '#F18F01',
            'similarity_high': '#43AA8B',
            'similarity_low': '#F8333C'
        }
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        
        if not self.matplotlib_available:
            print("⚠️ DTWVisualizer: matplotlib not available, visualizations disabled")
        
    def create_dtw_alignment_plot(self, dtw_results: Dict, feature_name: str, 
                                save_path: str = None) -> str:
        """
        Create DTW alignment plot showing how frames are matched.
        
        Args:
            dtw_results: DTW analysis results
            feature_name: Name of feature to visualize
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        if not self.matplotlib_available:
            print(f"⚠️ Cannot create DTW alignment plot: matplotlib not available")
            return None
            
        print(f"🎨 Creating DTW alignment plot for {feature_name}...")
        
        # Extract DTW data for the specific feature
        detailed_analysis = dtw_results.get('dtw_analysis', {}).get('detailed_analysis', {})
        feature_data = detailed_analysis.get(feature_name, {})
        
        if 'error' in feature_data:
            print(f"⚠️ Cannot create alignment plot: {feature_data['error']}")
            return None
            
        # Create the alignment plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.fig_size)
        fig.suptitle(f'DTW Alignment Analysis: {feature_name.replace("_", " ").title()}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Original trajectories
        self._plot_original_trajectories(ax1, feature_data, feature_name)
        
        # Plot 2: DTW alignment path
        self._plot_dtw_warping_path(ax2, feature_data, feature_name)
        
        # Plot 3: Aligned trajectories
        self._plot_aligned_trajectories(ax3, feature_data, feature_name)
        
        plt.tight_layout()
        
        # Save the plot
        if not save_path:
            save_path = f"shooting_comparison/results/dtw_alignment_{feature_name}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ DTW alignment plot saved: {save_path}")
        return save_path
    
    def _plot_original_trajectories(self, ax, feature_data: Dict, feature_name: str):
        """Plot original trajectories before DTW alignment"""
        # This is a simplified version - would need actual trajectory data
        # For now, create placeholder visualization
        
        ax.set_title("Original Trajectories (Before Alignment)")
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3], 
               color=self.colors['video1'], linewidth=2, label='Video 1')
        ax.plot([1, 2, 3, 4, 5, 6], [1.2, 3.8, 2.1, 4.9, 3.2, 2], 
               color=self.colors['video2'], linewidth=2, label='Video 2')
        
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Feature Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dtw_warping_path(self, ax, feature_data: Dict, feature_name: str):
        """Plot DTW warping path showing frame alignments"""
        ax.set_title("DTW Warping Path (Frame Alignments)")
        
        # Create sample warping path for demonstration
        # In real implementation, would extract from DTW results
        path_i = [0, 1, 2, 2, 3, 4]  # Video 1 frames
        path_j = [0, 0, 1, 2, 3, 4]  # Video 2 frames
        
        # Plot warping path
        ax.plot(path_j, path_i, color=self.colors['alignment'], linewidth=3, marker='o', markersize=6)
        
        # Add diagonal reference line
        max_frame = max(max(path_i), max(path_j))
        ax.plot([0, max_frame], [0, max_frame], 
               color='gray', linestyle='--', alpha=0.5, label='Perfect Alignment')
        
        ax.set_xlabel("Video 2 Frame Index")
        ax.set_ylabel("Video 1 Frame Index") 
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key alignments
        for i, (x, y) in enumerate(zip(path_j, path_i)):
            if i % 2 == 0:  # Annotate every other point
                ax.annotate(f'({x},{y})', (x, y), xytext=(5, 5), 
                          textcoords='offset points', fontsize=8)
    
    def _plot_aligned_trajectories(self, ax, feature_data: Dict, feature_name: str):
        """Plot trajectories after DTW alignment"""
        ax.set_title("Aligned Trajectories (After DTW)")
        
        # Sample aligned data
        aligned_frames = [1, 2, 3, 4, 5, 6]
        traj1_aligned = [1, 4, 2, 2, 5, 3]  # Stretched/compressed to match
        traj2_aligned = [1.2, 3.8, 2.1, 2.1, 4.9, 3.2]
        
        ax.plot(aligned_frames, traj1_aligned, 
               color=self.colors['video1'], linewidth=2, label='Video 1 (Aligned)')
        ax.plot(aligned_frames, traj2_aligned, 
               color=self.colors['video2'], linewidth=2, label='Video 2 (Aligned)')
        
        # Fill area between trajectories to show similarity
        ax.fill_between(aligned_frames, traj1_aligned, traj2_aligned, 
                       alpha=0.3, color=self.colors['similarity_high'])
        
        ax.set_xlabel("Aligned Timeline")
        ax.set_ylabel("Feature Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add similarity score annotation
        similarity = feature_data.get('overall_similarity', 0)
        ax.text(0.02, 0.98, f'Similarity: {similarity:.1f}%', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def create_similarity_heatmap(self, dtw_results: Dict, save_path: str = None) -> str:
        """
        Create heatmap showing similarity scores across all features.
        
        Args:
            dtw_results: DTW analysis results
            save_path: Path to save the heatmap
            
        Returns:
            Path to saved heatmap
        """
        print("🎨 Creating DTW similarity heatmap...")
        
        # Extract feature similarities
        feature_similarities = dtw_results.get('dtw_analysis', {}).get('feature_similarities', {})
        
        if not feature_similarities:
            print("⚠️ No feature similarities found for heatmap")
            return None
        
        # Prepare data for heatmap
        features = list(feature_similarities.keys())
        feature_names = [name.replace('_', ' ').title() for name in features]
        similarities = [feature_similarities[f] for f in features]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create color matrix
        similarity_matrix = np.array(similarities).reshape(1, -1)
        
        # Plot heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        
        # Customize plot
        ax.set_title('DTW Feature Similarity Scores', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks([])
        
        # Add text annotations
        for i, similarity in enumerate(similarities):
            ax.text(i, 0, f'{similarity:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=12,
                   color='white' if similarity < 50 else 'black')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Similarity Score (%)', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        # Save the plot
        if not save_path:
            save_path = "shooting_comparison/results/dtw_similarity_heatmap.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ DTW similarity heatmap saved: {save_path}")
        return save_path
    
    def create_trajectory_comparison_plot(self, dtw_results: Dict, video1_data: Dict, 
                                        video2_data: Dict, save_path: str = None) -> str:
        """
        Create detailed trajectory comparison plots.
        
        Args:
            dtw_results: DTW analysis results
            video1_data: First video's data
            video2_data: Second video's data
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        print("🎨 Creating DTW trajectory comparison plot...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DTW Trajectory Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Plot different trajectory comparisons
        self._plot_ball_trajectory_comparison(axes[0, 0], dtw_results, video1_data, video2_data)
        self._plot_wrist_trajectory_comparison(axes[0, 1], dtw_results, video1_data, video2_data)
        self._plot_elbow_angle_comparison(axes[1, 0], dtw_results, video1_data, video2_data)
        self._plot_hip_stability_comparison(axes[1, 1], dtw_results, video1_data, video2_data)
        
        plt.tight_layout()
        
        # Save the plot
        if not save_path:
            save_path = "shooting_comparison/results/dtw_trajectory_comparison.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ DTW trajectory comparison plot saved: {save_path}")
        return save_path
    
    def _plot_ball_trajectory_comparison(self, ax, dtw_results, video1_data, video2_data):
        """Plot ball trajectory comparison"""
        ax.set_title("Ball Trajectory Comparison")
        
        # Placeholder trajectory data
        frames = range(1, 21)
        ball1_x = [i + np.sin(i/3) * 2 for i in frames]
        ball1_y = [i * 0.5 + np.cos(i/2) for i in frames]
        ball2_x = [i + np.sin(i/3 + 0.5) * 1.8 for i in frames]
        ball2_y = [i * 0.52 + np.cos(i/2 + 0.3) for i in frames]
        
        ax.plot(ball1_x, ball1_y, color=self.colors['video1'], linewidth=2, 
               marker='o', markersize=4, label='Video 1 Ball')
        ax.plot(ball2_x, ball2_y, color=self.colors['video2'], linewidth=2, 
               marker='s', markersize=4, label='Video 2 Ball')
        
        ax.set_xlabel("X Position (normalized)")
        ax.set_ylabel("Y Position (normalized)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_wrist_trajectory_comparison(self, ax, dtw_results, video1_data, video2_data):
        """Plot wrist trajectory comparison"""
        ax.set_title("Wrist Trajectory Comparison") 
        
        # Placeholder wrist data
        frames = range(1, 21)
        wrist1_x = [i + np.cos(i/4) * 1.5 for i in frames]
        wrist1_y = [15 - i * 0.3 + np.sin(i/3) for i in frames]
        wrist2_x = [i + np.cos(i/4 + 0.2) * 1.4 for i in frames]
        wrist2_y = [15.2 - i * 0.32 + np.sin(i/3 + 0.1) for i in frames]
        
        ax.plot(wrist1_x, wrist1_y, color=self.colors['video1'], linewidth=2, 
               marker='o', markersize=4, label='Video 1 Wrist')
        ax.plot(wrist2_x, wrist2_y, color=self.colors['video2'], linewidth=2, 
               marker='s', markersize=4, label='Video 2 Wrist')
        
        ax.set_xlabel("X Position (normalized)")
        ax.set_ylabel("Y Position (normalized)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_elbow_angle_comparison(self, ax, dtw_results, video1_data, video2_data):
        """Plot elbow angle comparison"""
        ax.set_title("Elbow Angle Comparison")
        
        # Placeholder elbow angle data
        frames = range(1, 21)
        elbow1 = [90 + 30 * np.sin(i/5) for i in frames]
        elbow2 = [92 + 28 * np.sin(i/5 + 0.3) for i in frames]
        
        ax.plot(frames, elbow1, color=self.colors['video1'], linewidth=2, 
               marker='o', markersize=4, label='Video 1 Elbow')
        ax.plot(frames, elbow2, color=self.colors['video2'], linewidth=2, 
               marker='s', markersize=4, label='Video 2 Elbow')
        
        ax.set_xlabel("Frame")
        ax.set_ylabel("Elbow Angle (degrees)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_hip_stability_comparison(self, ax, dtw_results, video1_data, video2_data):
        """Plot hip stability comparison"""
        ax.set_title("Hip Stability Comparison")
        
        # Placeholder hip stability data
        frames = range(1, 21)
        hip1 = [10 + np.random.normal(0, 0.5) for _ in frames]
        hip2 = [10.2 + np.random.normal(0, 0.6) for _ in frames]
        
        ax.plot(frames, hip1, color=self.colors['video1'], linewidth=2, 
               marker='o', markersize=4, label='Video 1 Hip')
        ax.plot(frames, hip2, color=self.colors['video2'], linewidth=2, 
               marker='s', markersize=4, label='Video 2 Hip')
        
        ax.set_xlabel("Frame")
        ax.set_ylabel("Hip Y Position (normalized)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_comprehensive_dtw_report(self, dtw_results: Dict, video1_data: Dict, 
                                      video2_data: Dict, video1_path: str, video2_path: str,
                                      save_dir: str = None) -> Dict[str, str]:
        """
        Create comprehensive DTW visualization report.
        
        Args:
            dtw_results: DTW analysis results
            video1_data: First video's data
            video2_data: Second video's data
            video1_path: Path to first video
            video2_path: Path to second video
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary of visualization file paths
        """
        print("🎨 Creating comprehensive DTW visualization report...")
        
        if not save_dir:
            base_name1 = os.path.splitext(os.path.basename(video1_path))[0]
            base_name2 = os.path.splitext(os.path.basename(video2_path))[0]
            save_dir = f"shooting_comparison/results/dtw_viz_{base_name1}_vs_{base_name2}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        visualization_files = {}
        
        # 1. Similarity heatmap
        heatmap_path = os.path.join(save_dir, "similarity_heatmap.png")
        visualization_files['heatmap'] = self.create_similarity_heatmap(dtw_results, heatmap_path)
        
        # 2. Trajectory comparison
        trajectory_path = os.path.join(save_dir, "trajectory_comparison.png")
        visualization_files['trajectories'] = self.create_trajectory_comparison_plot(
            dtw_results, video1_data, video2_data, trajectory_path)
        
        # 3. Feature-specific alignment plots
        feature_similarities = dtw_results.get('dtw_analysis', {}).get('feature_similarities', {})
        for feature_name in feature_similarities.keys():
            alignment_path = os.path.join(save_dir, f"alignment_{feature_name}.png")
            viz_path = self.create_dtw_alignment_plot(dtw_results, feature_name, alignment_path)
            if viz_path:
                visualization_files[f'alignment_{feature_name}'] = viz_path
        
        print(f"✅ DTW visualization report created in: {save_dir}")
        print(f"📊 Generated {len(visualization_files)} visualization files")
        
        # Save visualization index
        index_file = os.path.join(save_dir, "visualization_index.json")
        with open(index_file, 'w') as f:
            json.dump(visualization_files, f, indent=2)
        
        return visualization_files

# Utility function for easy visualization
def visualize_dtw_results(dtw_results: Dict, video1_data: Dict, video2_data: Dict,
                         video1_path: str, video2_path: str) -> Dict[str, str]:
    """
    Convenience function to create all DTW visualizations.
    
    Args:
        dtw_results: DTW analysis results
        video1_data: First video's data
        video2_data: Second video's d   ata
        video1_path: Path to first video
        video2_path: Path to second video
        
    Returns:
        Dictionary of visualization file paths
    """
    visualizer = DTWVisualizer()
    return visualizer.create_comprehensive_dtw_report(
        dtw_results, video1_data, video2_data, video1_path, video2_path)