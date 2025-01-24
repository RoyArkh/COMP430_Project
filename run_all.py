import subprocess

def run_scripts(model_name, dataset_name):
    
    # Step 1: Run TestNeighbours.py
    print("########################################################")
    print("running neighbourhood")
    print("########################################################")
    subprocess.run([
        "python", "TestNeighbours.py",
        "--model", model_name,
        "--dataset", dataset_name
    ])
    
    
    print("########################################################")
    print("running other attacks")
    print("########################################################")
    # Step 2: Run mia_run.py
    subprocess.run([
        "python", "mia_run.py",
        "--root_dataset", "x_sum",
        "--dataset", "dataset_32",
        "--model", model_name
    ])
    
    
    print("########################################################")
    print("merging")
    print("########################################################")
    # Step 3: Run add_nb.py
    subprocess.run([
        "python", "add_nb.py",
        f"Neighbourhood/results_N50_wikitext/{model_name[8:]}_full_scores.csv",
        f"results/latest_wikitext/{model_name[8:]}_full_scores.csv",
        f"merged/{model_name[8:]}_full_scores.csv"
    ])
    

if __name__ == "__main__":
    # Specify names of models and dataset here
    models = []
    dataset = ""
    #run_scripts("mia-llm/pythia-70m-wikitext2raw-roya", wikitext_dataset)
    for model in models:
        run_scripts(model, dataset)
