import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
results_csv_path = "G:\\PDC-Proj\\Results\\Results.csv"
community_csv_path = "G:\\PDC-Proj\\Results\\Community.csv"

# Load CSVs
results_df = pd.read_csv(results_csv_path)
community_df = pd.read_csv(community_csv_path)

# Clean column names
results_df.columns = results_df.columns.str.strip()
community_df.columns = community_df.columns.str.strip()

print("Columns in results_df:", results_df.columns.tolist())
print(results_df.head())

# Clean time-related columns
for col in ['Total Time (s)', 'Average Community Contraction Time (s)']:
    if col in results_df.columns:
        results_df[col] = results_df[col].astype(str).str.replace('s', '', regex=False).astype(float)

# --- Plot 1: Total Time Comparison ---
plt.figure(figsize=(8, 6))
sns.barplot(x="Run Type", y="Total Time (s)", data=results_df, palette="viridis")
plt.title("Total Execution Time")
plt.ylabel("Time (s)")
plt.xlabel("Run Type")
plt.tight_layout()
plt.savefig("total_execution_time.png")

# --- Plot 2: Average Community Contraction Time ---
plt.figure(figsize=(8, 6))
sns.barplot(x="Run Type", y="Average Community Contraction Time (s)", data=results_df, palette="plasma")
plt.title("Avg Community Contraction Time")
plt.ylabel("Time (s)")
plt.xlabel("Run Type")
plt.tight_layout()
plt.savefig("avg_community_contraction_time.png")

# --- Plot 3: Community Count & Tensors per Community ---
fig, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x="Run Type", y="Communities Detected", data=results_df, palette="mako", ax=ax1)
ax1.set_ylabel("Communities Detected")
ax1.set_title("Communities and Avg Tensors per Community")
ax2 = ax1.twinx()
sns.lineplot(x="Run Type", y="Avg Tensors per Community", data=results_df, sort=False, marker='o', ax=ax2, color='red')
ax2.set_ylabel("Avg Tensors per Community")
plt.tight_layout()
plt.savefig("communities_vs_avg_tensors.png")

# --- Plot 4: Per-Community Contraction Times ---
community_df.columns = ["Run Type", "Community ID", "Tensors", "Contraction Time (s)"]
community_df["Contraction Time (s)"] = community_df["Contraction Time (s)"].astype(float)

plt.figure(figsize=(12, 8))
sns.barplot(x="Community ID", y="Contraction Time (s)", hue="Run Type", data=community_df, palette="Set2")
plt.title("Per-Community Contraction Times")
plt.xlabel("Community ID")
plt.ylabel("Time (s)")
plt.legend(title="Run Type")
plt.tight_layout()
plt.savefig("per_community_contraction_times.png")
