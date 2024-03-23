from datasets import load_dataset

all_corpus = ["MedQA","MedMCQA"]
all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

test_cost = []

for corpus_n in all_corpus:

    dataset = load_dataset("SpeechLLM/BioInstructQA", corpus_n)

    print(corpus_n)

    longer = []
    total = []

    for d in dataset["test"]:

        if len(d["prompt_no_answer"]) > 4096:
            longer.append(len(d["prompt_no_answer"]))
        total.append(len(d["prompt_no_answer"]))
    
    print(len(longer), " => ", longer)
    print("Sum: ", sum(total))
    cost = (sum(total) / 2117025) * 33.14
    print("Cost: ", cost)
    test_cost.append(cost)

print("Total test cost: ", sum(test_cost))

print("#"*50)    
print("# Train")    
print("#"*50)    

for corpus_n in ["MedQA","MedMCQA"]:

    dataset = load_dataset("SpeechLLM/BioInstructQA", corpus_n)

    print(corpus_n)

    longer = []
    total = []

    for d in dataset["train"]:

        if len(d["prompt_no_answer"]) > 4096:
            longer.append(len(d["prompt_no_answer"]))
        total.append(len(d["prompt_no_answer"]))
    
    print(len(longer), " => ", longer)
    print("Sum: ", sum(total))
    print("Cost: ", (sum(total) / 2117025) * 33.14)
    