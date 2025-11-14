import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def central(x):
    s = pd.Series(x).dropna()
    mean_val = s.mean()
    median_val = s.median()

    modes = s.mode()
    mode_val = modes.iloc[0] if len(modes) > 0 else np.nan

    return pd.Series(
        [mean_val, median_val, mode_val],
        index=["mean", "median", "mode"], )


def dispersion(x):
  
    s = pd.Series(x).dropna()

    std_val = s.std()
    min_val = s.min()
    max_val = s.max()
    range_val = max_val - min_val
    q25 = s.quantile(0.25)
    q75 = s.quantile(0.75)
    iqr = q75 - q25

    return pd.Series(
        [std_val, min_val, max_val, range_val, q25, q75, iqr],
        index=["std", "min", "max", "range", "25th", "75th", "IQR"],
    )


def corrcoef(x, y):
   
    return np.corrcoef(x, y)[0, 1]


def plot_regression_line(ax, x, y, **kwargs):
   
    x = np.asarray(x)
    y = np.asarray(y)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return

    # y = m * x + b
    m, b = np.polyfit(x, y, 1)

    xx = np.linspace(x.min(), x.max(), 100)
    yy = m * xx + b
    ax.plot(xx, yy, **kwargs)



def display_summary_table(df):
  
    cols = ["Anxiety", "Age", "Hours per day", "BPM"]
    num_df = df[cols].copy()

    central_tbl = num_df.apply(central, axis=0)
    dispersion_tbl = num_df.apply(dispersion, axis=0)

    print("=== Central tendency summary statistics ===")
    print(central_tbl.round(3))
    print()
    print("=== Dispersion summary statistics ===")
    print(dispersion_tbl.round(3))

    return central_tbl, dispersion_tbl




def plot_descriptive(df):
    y = df["Anxiety"].values
    age = df["Age"].values
    duration = df["Duration"].values
    bpm = df["BPM"].values

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), tight_layout=True)
    axes_flat = axes.ravel()

    ivs = [age, duration, bpm]
    xlabels = ["Age", "Duration", "BPM"]
    colors = ["b", "r", "g"]

    for ax, x, lab, c in zip(axes_flat[:3], ivs, xlabels, colors):
        ax.scatter(x, y, alpha=0.5, color=c)
        plot_regression_line(ax, x, y, color="k", ls="-", lw=2)

        r = corrcoef(x, y)
        ax.text(
            0.7, 0.85, f"r = {r:.3f}",
            transform=ax.transAxes,
            bbox=dict(fc="0.9", alpha=0.7),
        )

        ax.set_xlabel(lab)
        ax.set_ylabel("Anxiety")


    axes_flat[0].set_xticks(np.arange(10, 90, 20))


    ax = axes_flat[3]

    young = age < 30
    old = age >= 30

    colors_group = ["m", "c"]
    labels_group = ["Age < 30", "Age ≥ 30"]
    ylocs = [0.8, 0.7]

    for mask, c, label, yloc in zip(
        [young, old], colors_group, labels_group, ylocs
    ):
        ax.scatter(duration[mask], y[mask], alpha=0.5, color=c, label=label)
        plot_regression_line(ax, duration[mask], y[mask], color="k", ls="-", lw=2)

        r_group = corrcoef(duration[mask], y[mask])
        ax.text(
            0.05, yloc, f"r = {r_group:.3f}",
            transform=ax.transAxes,
            bbox=dict(fc="0.9", alpha=0.8),
            color=c,
        )

    ax.set_xlabel("Duration")
    ax.set_ylabel("Anxiety")
    ax.legend()

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for ax, lab in zip(axes_flat, panel_labels):
        ax.text(0.02, 0.95, lab, transform=ax.transAxes)

    return fig, axes


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python descriptive.py path_to_data.csv")
        sys.exit(0)

    path = sys.argv[1]
    data = pd.read_csv(path)

    cols = ["Anxiety", "Age", "Duration", "BPM"]
    data = data[cols]

    display_summary_table(data)
    plot_descriptive(data)
    plt.show()

