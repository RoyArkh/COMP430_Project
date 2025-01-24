import subprocess

def run_scripts(model_name, dataset_name):
    
    # Step 1: Run Neighbourhood/TestNeighbours.py
    print("running neighbourhood")
    subprocess.run([
        "python", "Neighbourhood/TestNeighbours.py",
        "--model", model_name,
        "--dataset", dataset_name
    ])
    
    
    print("running other attacks")
    # Step 2: Run mia_run.py
    subprocess.run([
        "python", "mia_run.py",
        "--root_dataset", "x_sum",
        "--dataset", "dataset_32",
        "--model", model_name
    ])
    
    print("merging")
    # Step 3: Run add_nb.py
    subprocess.run([
        "python", "add_nb.py",
        f"Neighbourhood/results_N50_xsum/{model_name[8:]}_full_scores.csv",
        f"results/latest_xsum/{model_name[8:]}_full_scores.csv",
        f"merged/{model_name[8:]}_full_scores.csv"
    ])
    

if __name__ == "__main__":
    # Specify NameOfModel and NameOfDataset here
    pythia_models_wikitext = ["mia-llm/pythia-160m-AGnews-roya",
                        "mia-llm/pythia-410m-AGnews-roya",
                        "mia-llm/pythia-1b-AGnews-roya",
                        "mia-llm/pythia-1.4b-wikitext2raw-roya"]
    
    gptneo_models_wikitext = ["mia-llm/gpt-neo-125m-AGnews-roya", "mia-llm/gpt-neo-1.3B-AGnews-roya"]

    mobilellm_models_wikitext = ["mia-llm/MobileLLM-350M-AGnews-roya", "mia-llm/MobileLLM-600M-AGnews-roya", "mia-llm/MobileLLM-1B-AGnews-roya", "mia-llm/MobileLLM-1.5B-AGnews-roya"]
    #name_of_model = "mia-llm/pythia-70m-wikitext2raw-roya"
    #wikitext_dataset = "h0ssn/Agnews_MIA_benchmark_RH"

    pythia_models_xsum = [#"mia-llm/pythia-70m-xsum-roya",
                            "mia-llm/pythia-160m-xsum-roya", 
                            "mia-llm/pythia-410m-xsum-roya",
                            "mia-llm/pythia-1b-xsum-roya"]

    gptneo_models_wikitext=["mia-llm/gpt-neo-125m-xsum-roya", "mia-llm/gpt-neo-1.3B-xsum-roya"]

    mobilellm_models_wikitext = ["mia-llm/MobileLLM-350M-xsum-roya", "mia-llm/MobileLLM-600M-xsum-roya", "mia-llm/MobileLLM-1B-xsum-roya", "mia-llm/MobileLLM-1.5B-xsum-roya"]

    agnews_dataset = "h0ssn/Agnews_MIA_benchmark_RH"
    xsum_dataset = "h0ssn/Xsum_MIA_benchmark_RH"
    
    #run_scripts("mia-llm/pythia-70m-xsum-roya", xsum_dataset)
    for model in gptneo_models_wikitext:
        run_scripts(model, xsum_dataset)
