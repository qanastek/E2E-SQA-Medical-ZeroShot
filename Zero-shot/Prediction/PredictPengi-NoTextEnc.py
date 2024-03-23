import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from wrapper import PengiWrapper as Pengi
from sklearn.metrics import classification_report, accuracy_score

pengi = Pengi(config="base_no_text_enc")

all_corpus = ["MedQA","MedMCQA"]
all_corpus = ["MMLU_" + subject for subject in ["clinical_knowledge","medical_genetics","anatomy","professional_medicine","college_biology","college_medicine"]] + all_corpus

resume_corpus = {}

for corpus in all_corpus:

    dataset = load_dataset("SpeechLLM/SpokenMedicalQA", corpus)["test"]
    print(dataset)

    results = []

    with torch.no_grad():
        
        for d in tqdm(dataset):
                    
            audio_file_paths = [d["wav_no_answer"]]

            text_prompts = ["The correct answer is option "]
            add_texts = [""]

            generated_response = pengi.generate(
                audio_paths=audio_file_paths,
                text_prompts=text_prompts,
                add_texts=add_texts,
                max_len=1,
                beam_size=50257,
                # beam_size=2,
                temperature=1.0,
                stop_token=' <|endoftext|>',
            )
            res = generated_response[0]
            tokens = res[0]
            probabilities = res[1].tolist()

            letters_probs = []

            for letter in ["A","B","C","D"]:
                index_letter = tokens.index(letter)
                letter_prob = probabilities[index_letter]
                letters_probs.append((letter, letter_prob))

            print(letters_probs)

            probs = [lp[1] for lp in letters_probs]
            best_prob = min(probs)
            index_best_prob = probs.index(best_prob)
            best_letter = letters_probs[index_best_prob][0]

            print(best_letter)
            print(d["class"])
            print()

            results.append({
                "identifier": d["ID"],
                "identifier": d["ID_original_dataset"],
                
                "probabilities": letters_probs,

                "best_letter": best_letter,
                "expected_letter": d["class"],
            })

    f_out = open(f"./results_pengi/results_Pengi_ZeroShot_{corpus}_EN_base_no_text_enc.json", 'w')
    json.dump(results, f_out)
    f_out.close()

    refs = [r["expected_letter"] for r in results]
    preds = [r["best_letter"] for r in results]

    acc = accuracy_score(refs, preds)
    print("Accuracy: ", acc)

    resume_corpus[corpus] = acc

f_out_resume = open(f"./results_pengi/results_Pengi_ZeroShot_Resume_EN_base_no_text_enc.json", 'w')
json.dump(resume_corpus, f_out_resume)
f_out_resume.close()
