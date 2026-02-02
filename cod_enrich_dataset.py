import pandas as pd
from cod_prefilter import run_cod_llm  # import your CoD function

def main():
    # 1. Load your manually labeled dataset
    df = pd.read_csv(r"C:\Users\anany\Downloads\custom_prompt_ds.csv")  # adjust path if needed

    # 2. Prepare new columns
    draft_analyses = []
    draft_labels = []

    # 3. Run CoD on each prompt
    for _, row in df.iterrows():
        prompt = row["prompt"]
        da, dl = run_cod_llm(prompt)
        draft_analyses.append(da)
        draft_labels.append(dl)

    # 4. Add CoD outputs to the DataFrame
    df["draft_analysis"] = draft_analyses
    df["draft_label"] = draft_labels

    # 5. Rename label to true_label if you want
    df = df.rename(columns={"label": "true_label"})

    # 6. Save enriched CSV
    df.to_csv(r"C:\hybrid_prompt_detection\cod_enriched_dataset.csv", index=False)
    print("Saved data/cod_enriched_dataset.csv")

if __name__ == "__main__":
    main()
 