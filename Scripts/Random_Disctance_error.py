import numpy as np
import laspy
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import norm

def load_las_points(filepath):
    las = laspy.read(filepath)
    return np.vstack((las.x, las.y, las.z)).T

def estimate_plane_pca(points):
    centroid = points.mean(axis=0)
    centered = points - centroid
    pca = PCA(n_components=3).fit(centered)
    normal = pca.components_[-1]
    d = -np.dot(normal, centroid)
    return normal, d, centroid, pca

def compute_distances(points, normal, d):
    return (points @ normal + d) / np.linalg.norm(normal)

def compute_beam_residuals(points, normal, d):
    R_meas = np.linalg.norm(points, axis=1)
    b_dirs = points / R_meas[:, None]
    denom = b_dirs @ normal
    R_ideal = -d / denom
    return R_meas - R_ideal

def save_radial_histogram(residuals, output_path):
    residuals_cm = residuals * 100
    mean, std = residuals_cm.mean(), residuals_cm.std()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals_cm, bins=50, density=True, color='#219EBC', edgecolor='white', label='Measured Data')

    x = np.linspace(residuals_cm.min(), residuals_cm.max(), 500)
    plt.plot(x, norm.pdf(x, mean, std), color='#023047', label=f'Gaussian Fit (μ = {mean:.2f}, σ = {std:.2f} cm)')
    for sign in [-1, 1]:
        plt.axvline(mean + sign * std, color='#FB8500', linestyle='--', label='±1σ' if sign == 1 else None)

    plt.xlabel('Radial Error (cm, centered)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.title('Beam‐Direction Residual Distribution (1σ)', fontsize=20)
    plt.xticks( fontsize=16 )
    plt.yticks( fontsize=16)
    plt.legend( fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_3d_plot(points, normal, centroid, pca, output_path):
    u, v = pca.components_[:2]
    extent = np.percentile(np.linalg.norm(points - centroid, axis=1), 95)
    grid = np.linspace(-extent, extent, 10)
    uu, vv = np.meshgrid(grid, grid)
    plane = centroid[:, None, None] + uu[None] * u[:, None, None] + vv[None] * v[:, None, None]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(*plane, color="#219EBC", alpha=0.8, shade=False, label='PCA Fitted Plane')
    ax.scatter(*points.T, s=1, color='#023047', label ='Points')

    for axis, label in zip("xyz", "XYZ"):
        getattr(ax, f"set_{axis}label")(label)

    all_pts = np.vstack((points, plane.reshape(3, -1).T))
    mid = all_pts.mean(axis=0)
    max_range = np.ptp(all_pts, axis=0).max() / 2
    limits = [(m - max_range, m + max_range) for m in mid]

    ax.set_title("3D Point Cloud and Estimated Plane", fontsize=16)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.view_init(elev=25, azim=89)
    plt.xticks( fontsize=14)
    plt.yticks( fontsize=14)
    plt.legend( fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_file(filepath, output_dir):
    file_name = os.path.splitext(os.path.basename(filepath))[0]
    points = load_las_points(filepath)
    print(f"\nProcessing '{file_name}.las' with {points.shape[0]} points")

    normal, d, centroid, pca = estimate_plane_pca(points)
    inliers = points[np.abs(compute_distances(points, normal, d)) < 1]
    normal, d, centroid, pca = estimate_plane_pca(inliers)

    residuals = compute_beam_residuals(inliers, normal, d)
    mean_res, std_res = residuals.mean(), residuals.std()
    scanner_dist = abs(d) / np.linalg.norm(normal)

    hist_path = os.path.join(output_dir, f"{file_name}_radial_hist.png")
    plot_path = os.path.join(output_dir, f"{file_name}_3d.png")
    save_radial_histogram(residuals, hist_path)
    save_3d_plot(inliers, normal, centroid, pca, plot_path)

    summary_path = os.path.join(output_dir, f"{file_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("--- Plane & Measurement Summary ---\n")
        f.write(f"File: {file_name}.las\n")
        f.write(f"Inlier points (<0.1m): {inliers.shape[0]}\n")
        f.write(f"Scanner-to-plane distance: {scanner_dist:.2f} m\n")
        f.write(f"Radial error (1σ): {std_res * 100:.2f} cm\n")
        f.write(f"Mean radial bias: {mean_res * 100:.2f} cm\n")

    print(f"Saved radial histogram: {hist_path}")
    print(f"Saved 3D plot:          {plot_path}")
    print(f"Saved summary:          {summary_path}")

    #return metrics for global summary
    return {"file": file_name, "inliers": inliers.shape[0], "scanner_dist_m": scanner_dist, "radial_error_cm": std_res * 100, "mean_bias_cm": mean_res * 100,}

def main():
    parser = argparse.ArgumentParser(description="Process all .las files in a directory.")
    parser.add_argument("input_folder", help="Path to the folder containing .las files.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a valid directory.")
        return

    las_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(".las")]
    if not las_files:
        print("No .las files found in the directory.")
        return

    all_results = []
    for las_file in las_files:
        result = process_file(os.path.join(args.input_folder, las_file), args.input_folder)
        all_results.append(result)

    # Save overall summary
    overall_path = os.path.join(args.input_folder, "overall_summary.csv")
    with open(overall_path, "w", encoding="utf-8") as f:
        f.write("file,inliers,scanner_dist_m,radial_error_cm,mean_bias_cm\n")
        for res in all_results:
            f.write(f"{res['file']},{res['inliers']},{res['scanner_dist_m']:.2f},"
                    f"{res['radial_error_cm']:.2f},{res['mean_bias_cm']:.2f}\n")

    print(f"\nSaved overall summary: {overall_path}")


if __name__ == "__main__":
    main()

#To run put the cutout .las files in a single folder, type python Random_Distance_error.py [folder name]
#This will start the tool and calculate, plot and povide a summary of the RDE estimation for each cutout .las.
#Finally it will provide a final summary in a .csv file showing the results of all files in the folder.
#All outputs are saved in the same folder as where the .las files are taken from