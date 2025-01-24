# COMP430_Project
SLM MIA Benchmark

To run the code on different fine-tuned models define an array in the run_all.py file and put down the name of the dataset you want to use for the attack.

```python
# Specify names of models and dataset here
models = []
dataset = ""
#example of how it should look like: run_scripts("mia-llm/pythia-70m-wikitext2raw-roya", wikitext_dataset)
for model in models:
    run_scripts(model, dataset)
```

p.s. make sure to change the saving and loading directories for it to work properly
