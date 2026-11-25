import pandas as pd
import matplotlib.pyplot as plt

datasets = {
    "MIT-BIH": pd.read_csv("mitbih_results.csv"),
    "TwoLeadECG": pd.read_csv("twolead_results.csv")
}

markers = {
    "SVM": "o",
    "Random Forest": "s",
    "XGBoost": "^"
}

colors = {
    "Raw": "#7f7f7f",
    "PE": "#9467bd",
    "BC": "#1f77b4",
    "PL": "#ff7f0e",
    "PI": "#2ca02c",
    "BCT": "#1f77b4",
    "PLT": "#ff7f0e",
    "PIT": "#2ca02c",
    "PET": "#d62728"
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (dataset_name, df) in zip(axes, datasets.items()):
    for _, row in df.iterrows():
        rep = row["Representation"]
        model = row["Model"]

        ax.scatter(
            row["Dimension"],
            row["AUC Mean"],
            color=colors[rep],
            marker=markers[model],
            s=130 if rep == "PET" else 65,
            edgecolor="black",
            linewidth=0.5,
            zorder=5 if rep == "PET" else 2
        )

    for rep, group in df.groupby("Representation"):
        best = group.loc[group["AUC Mean"].idxmax()]

        ax.annotate(
            rep,
            (best["Dimension"], best["AUC Mean"]),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold" if rep == "PET" else "normal"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Feature dimensionality")
    ax.set_ylabel("ROC-AUC")
    ax.set_title(dataset_name)
    ax.grid(alpha=0.2)

handles = [
    plt.Line2D(
        [0], [0],
        marker=marker,
        color="black",
        linestyle="None",
        label=model,
        markersize=7
    )
    for model, marker in markers.items()
]

fig.legend(handles=handles, loc="upper center", ncol=3)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig("performance_dimensionality.pdf", bbox_inches="tight")
plt.show()