#!/usr/bin/env python3
"""
Parse Kaggle notebooks CSV and generate markdown for high-voted notebooks.
"""

import csv
from datetime import datetime


def parse_notebooks_csv(csv_file, min_votes=10):
    """Parse CSV file and filter notebooks with minimum votes."""
    notebooks = []

    with open(csv_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            votes = int(row["totalVotes"])
            if votes >= min_votes:
                # Create notebook URL
                notebook_url = f"https://www.kaggle.com/code/{row['ref']}"

                # Parse date for better formatting
                try:
                    last_run = datetime.fromisoformat(row["lastRunTime"].replace("000", ""))
                    formatted_date = last_run.strftime("%Y-%m-%d")
                except:
                    formatted_date = row["lastRunTime"].split(" ")[0]

                notebooks.append(
                    {
                        "title": row["title"],
                        "author": row["author"],
                        "votes": votes,
                        "url": notebook_url,
                        "ref": row["ref"],
                        "last_run": formatted_date,
                    }
                )

    # Sort by votes (descending)
    notebooks.sort(key=lambda x: x["votes"], reverse=True)
    return notebooks


def generate_markdown(notebooks, output_file):
    """Generate markdown file with notebook list."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# CMI Competition: High-Voted Notebooks (10+ votes)\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total notebooks with 10+ votes: **{len(notebooks)}**\n\n")

        # Summary statistics
        vote_counts = [nb["votes"] for nb in notebooks]
        f.write("## Summary Statistics\n")
        f.write(f"- **Highest votes**: {max(vote_counts)}\n")
        f.write(f"- **Average votes**: {sum(vote_counts) / len(vote_counts):.1f}\n")
        f.write(f"- **Notebooks with 50+ votes**: {len([v for v in vote_counts if v >= 50])}\n")
        f.write(f"- **Notebooks with 100+ votes**: {len([v for v in vote_counts if v >= 100])}\n\n")

        # Top performers section
        f.write("## 🏆 Top Performers (100+ votes)\n\n")
        top_notebooks = [nb for nb in notebooks if nb["votes"] >= 100]

        if top_notebooks:
            f.write("| Rank | Title | Author | Votes | Last Updated |\n")
            f.write("|------|-------|--------|-------|-------------|\n")
            for i, nb in enumerate(top_notebooks, 1):
                f.write(f"| {i} | [{nb['title']}]({nb['url']}) | {nb['author']} | {nb['votes']} | {nb['last_run']} |\n")

        f.write("\n## 📊 All High-Voted Notebooks\n\n")
        f.write("| Rank | Title | Author | Votes | Last Updated |\n")
        f.write("|------|-------|--------|-------|-------------|\n")

        for i, nb in enumerate(notebooks, 1):
            f.write(f"| {i} | [{nb['title']}]({nb['url']}) | {nb['author']} | {nb['votes']} | {nb['last_run']} |\n")

        # Categories section
        f.write("\n## 📈 Notable Categories\n\n")

        # EDA notebooks
        eda_notebooks = [
            nb
            for nb in notebooks
            if any(keyword in nb["title"].lower() for keyword in ["eda", "visualization", "viz", "analysis"])
        ]
        if eda_notebooks:
            f.write("### 🔍 EDA & Visualization\n")
            for nb in eda_notebooks[:10]:  # Top 10
                f.write(f"- [{nb['title']}]({nb['url']}) by {nb['author']} ({nb['votes']} votes)\n")
            f.write("\n")

        # Baseline/Solution notebooks
        baseline_notebooks = [
            nb
            for nb in notebooks
            if any(keyword in nb["title"].lower() for keyword in ["baseline", "solution", "model"])
        ]
        if baseline_notebooks:
            f.write("### ⚡ Baselines & Solutions\n")
            for nb in baseline_notebooks[:10]:  # Top 10
                f.write(f"- [{nb['title']}]({nb['url']}) by {nb['author']} ({nb['votes']} votes)\n")
            f.write("\n")

        # High LB score notebooks
        lb_notebooks = [
            nb
            for nb in notebooks
            if "lb" in nb["title"].lower() and any(score in nb["title"].lower() for score in ["0.7", "0.8"])
        ]
        if lb_notebooks:
            f.write("### 🎯 High LB Scores\n")
            for nb in lb_notebooks[:10]:  # Top 10
                f.write(f"- [{nb['title']}]({nb['url']}) by {nb['author']} ({nb['votes']} votes)\n")
            f.write("\n")

        f.write("\n---\n")
        f.write(
            "*Data sourced from Kaggle API using `kaggle kernels list --competition cmi-detect-behavior-with-sensor-data --sort-by voteCount`*\n"
        )


if __name__ == "__main__":
    # Parse notebooks
    notebooks = parse_notebooks_csv("../../cmi_notebooks.csv", min_votes=10)

    # Generate markdown
    generate_markdown(notebooks, "../../docs/high_voted_cmi_notebooks.md")

    print(f"✅ Successfully generated markdown file with {len(notebooks)} notebooks")
    print("📁 Output file: high_voted_cmi_notebooks.md")
