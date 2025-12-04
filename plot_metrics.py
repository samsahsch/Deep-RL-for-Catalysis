import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# --- Configuration ---
# The columns we want to plot (Validation RMSE is the standard metric)
ENERGY_RMSE_COL = 'val0_epoch/per_atom_energy_rmse'
FORCES_RMSE_COL = 'val0_epoch/forces_rmse'
LOSS_COL = 'val0_epoch/weighted_sum'
EPOCH_COL = 'epoch'
FILENAME = 'metrics_gs.csv' # Changed to original default for consistency

def plot_training_history(filename=FILENAME):
    """
    Loads NequIP training metrics from a CSV file, cleans up unnecessary rows,
    and plots the most important error metrics (RMSE for Energy and Forces)
    on the validation set.
    """
    df = pd.read_csv(filename)

    # 1. Data Cleaning and Preprocessing
    # Filter for rows where validation metrics were logged (these are the end-of-epoch steps)
    # and drop duplicates from restarts if they exist.
    df_plot = df[df[ENERGY_RMSE_COL].notna()].copy()

    if df_plot.empty:
        print("Error: No validation data found. Check if the epoch columns are named correctly.")
        print("Expected columns:", [ENERGY_RMSE_COL, FORCES_RMSE_COL, LOSS_COL])
        sys.exit(1)

    # Ensure data types are numeric
    for col in [ENERGY_RMSE_COL, FORCES_RMSE_COL, LOSS_COL, EPOCH_COL]:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    # Remove any NaN entries that resulted from coercion
    df_plot.dropna(subset=[ENERGY_RMSE_COL, FORCES_RMSE_COL], inplace=True)
    
    # 2. Plotting
    
    # Calculate the epoch range
    max_epoch = df_plot[EPOCH_COL].max()
    
    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Allegro Training History for gs", fontsize=16)

    # --- Plot 1: Energy RMSE ---
    ax1 = axes[0]
    ax1.plot(df_plot[EPOCH_COL], df_plot[ENERGY_RMSE_COL] * 1000, label='Energy RMSE', linewidth=2) # Plot only raw data
    
    # Find the minimum energy RMSE and epoch (using the raw data now)
    min_e_rmse = df_plot[ENERGY_RMSE_COL].min() * 1000 
    min_e_epoch = df_plot.loc[df_plot[ENERGY_RMSE_COL].idxmin(), EPOCH_COL]
    
    ax1.scatter(min_e_epoch, min_e_rmse, color='red', marker='o', zorder=5, 
                label=f'Best: {min_e_rmse:.3f} (Epoch {min_e_epoch})')

    ax1.set_title('Validation Energy RMSE per Atom (meV/atom)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE (meV/atom)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.set_yscale('log')

    # --- Plot 2: Forces RMSE ---
    ax2 = axes[1]
    ax2.plot(df_plot[EPOCH_COL], df_plot[FORCES_RMSE_COL] * 1000, label='Forces RMSE', linewidth=2) # Plot only raw data
    
    # Find the minimum forces RMSE and epoch (using the raw data now)
    min_f_rmse = df_plot[FORCES_RMSE_COL].min() * 1000
    min_f_epoch = df_plot.loc[df_plot[FORCES_RMSE_COL].idxmin(), EPOCH_COL]
    
    ax2.scatter(min_f_epoch, min_f_rmse, color='red', marker='o', zorder=5, 
                label=f'Best: {min_f_rmse:.3f} (Epoch {min_f_epoch})')

    ax2.set_title(r'Validation Forces RMSE ($\rm meV/\AA$)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'RMSE ($\rm meV/\AA$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.show()
    
    print("\n--- Summary of Best Validation Performance (Raw Data) ---")
    print(f"Energy RMSE (Best): {min_e_rmse:.4f} eV/atom at Epoch {min_e_epoch}")
    print(f"Forces RMSE (Best): {min_f_rmse:.4f} eV/Å at Epoch {min_f_epoch}")


if __name__ == '__main__':
    plot_training_history()