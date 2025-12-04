import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

# --- Configuration ---
# File containing the JSON log entries
FILENAME = './mace_10e/results/MACE_model_run-123_train.txt' 

# Metrics to plot (corresponding to keys in the 'eval' mode logs)
ENERGY_RMSE_KEY = 'rmse_e_per_atom'
FORCES_RMSE_KEY = 'rmse_f'
EPOCH_KEY = 'epoch'

def load_and_plot_metrics(filename=FILENAME):
    """
    Loads JSON log entries from a file, filters for evaluation steps,
    processes the data, and plots the Energy and Forces RMSE.
    """
    log_entries = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                log_entries.append(json.loads(line))

    df = pd.DataFrame(log_entries)
    
    # Filter for rows where evaluation metrics were logged (mode == 'eval')
    df_plot = df[df['mode'] == 'eval'].copy()


    # Map the epoch column to correctly represent the completed epoch number.
    # 'epoch: null' (initial eval) is mapped to Epoch 0.
    # 'epoch: N' is mapped to Epoch N+1 (evaluation after N+1 training epochs).
    df_plot['epoch_corrected'] = df_plot[EPOCH_KEY].fillna(-1).astype(float) + 1
    df_plot['epoch_corrected'] = df_plot['epoch_corrected'].astype(int)

    # Convert units to meV (multiply by 1000)
    df_plot['energy_rmse_meV'] = df_plot[ENERGY_RMSE_KEY] * 1000
    df_plot['forces_rmse_meV'] = df_plot[FORCES_RMSE_KEY] * 1000
    
    # Drop rows where critical metrics are missing after conversion
    df_plot.dropna(subset=['energy_rmse_meV', 'forces_rmse_meV'], inplace=True)

    # Find the minimum RMSE values and their corresponding corrected epochs
    min_e_idx = df_plot['energy_rmse_meV'].idxmin()
    min_e_rmse = df_plot.loc[min_e_idx, 'energy_rmse_meV']
    min_e_epoch = df_plot.loc[min_e_idx, 'epoch_corrected']

    min_f_idx = df_plot['forces_rmse_meV'].idxmin()
    min_f_rmse = df_plot.loc[min_f_idx, 'forces_rmse_meV']
    min_f_epoch = df_plot.loc[min_f_idx, 'epoch_corrected']

    # Create the figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.plot(df_plot['epoch_corrected'], df_plot['energy_rmse_meV'], label='Energy RMSE', linewidth=2)

    ax1.scatter(min_e_epoch, min_e_rmse, color='red', marker='o', zorder=5, 
                label=f'Best: {min_e_rmse:.3f} (Epoch {min_e_epoch})')

    ax1.set_title('Validation Energy RMSE per Atom (meV/atom)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE (meV/atom)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    ax1.set_yscale('log')

    ax2 = axes[1]
    ax2.plot(df_plot['epoch_corrected'], df_plot['forces_rmse_meV'], label='Forces RMSE', linewidth=2)
    
    ax2.scatter(min_f_epoch, min_f_rmse, color='red', marker='o', zorder=5, 
                label=f'Best: {min_f_rmse:.3f} (Epoch {min_f_epoch})')

    ax2.set_title(r'Validation Forces RMSE ($\rm meV/\AA$)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'RMSE ($\rm meV/\AA$)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    print("\n--- Summary of Best Validation Performance ---")
    print(f"Energy RMSE (Best): {min_e_rmse:.4f} meV/atom at Epoch {min_e_epoch}")
    print(f"Forces RMSE (Best): {min_f_rmse:.4f} meV/Å at Epoch {min_f_epoch}")


if __name__ == '__main__':
    load_and_plot_metrics()