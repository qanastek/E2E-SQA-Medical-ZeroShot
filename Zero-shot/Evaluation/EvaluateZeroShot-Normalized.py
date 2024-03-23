import os
import json
import statistics

from sklearn.metrics import classification_report, accuracy_score

models_results = {}

bad_corpus_names = ["MMLU_" + m for m in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + ["MedQA","MedMCQA"]
good_corpus_name = [gcn.replace("_","-") for gcn in bad_corpus_names]

for allowed_model in ["References","small","medium","large-v2"]:

    print("="*50)
    print(allowed_model)

    for file_name in os.listdir("./results_zeroshot_normalized/"):

        new_file_name = file_name
        for mmlu_c in bad_corpus_names:
            new_file_name = new_file_name.replace(mmlu_c, mmlu_c.replace("_","-"))
        splitted = new_file_name.replace(".json","").split("_")
        model_name = splitted[1]
        shot_mode = splitted[2]
        corpus = splitted[3]
        lang = splitted[4]
        whisper_model = splitted[5]

        if whisper_model != allowed_model:
            continue

        # print(file_name)
        # print(splitted)

        f = open(f"./results_zeroshot_normalized/{file_name}")
        data = json.load(f)
        f.close()

        refs = []
        preds = []

        for d in data:
            refs.append(d["correct_letter"])
            list_predictions = list(d["predictions"].values())
            # print(list_predictions)
            max_value = max(list_predictions)
            # print(max_value)
            best_index = list_predictions.index(max_value)
            # print(best_index)
            best_prediction = list(d["predictions"].keys())[best_index]
            # print(best_prediction)
            preds.append(best_prediction)

        # cr = classification_report(refs, preds)
        # print(cr)

        acc = accuracy_score(refs, preds)
        # print(acc)

        if model_name not in models_results:
            models_results[model_name] = {}

        models_results[model_name][corpus] = acc

    line = " & " + " & ".join([m_name.replace("-"," ") for m_name in good_corpus_name]) + " \\\\"
    print(line)

    for model_name in sorted(list(models_results.keys())):

        values_out = [(corpus_name, models_results[model_name][corpus_name]*100) if corpus_name in models_results[model_name] else (corpus_name, 0) for corpus_name in good_corpus_name]
        values_out_mmlu = [v for c, v in values_out if "MMLU" in c]
        # avg_mmlu = sum(values_out_mmlu) / len(values_out_mmlu)

        formatted_values_out = ["{:.1f}".format(v) for c, v in values_out]
        average = sum([v for c, v in values_out]) / len(values_out)

        line = f"{model_name} & " + " & ".join(formatted_values_out) + " &   " + "{:.1f}".format(average) + " \\\\"
        print(line)
