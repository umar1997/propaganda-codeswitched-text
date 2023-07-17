import os
import re
import json
import time
from tqdm import tqdm
import openai
import pandas as pd
import numpy as np

from log import get_logger
from Model.evaluation import *
from dataPreparation import Dataset_Preparation

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
        
    )
    return response.choices[0].message["content"]

# To add X-shot add X number of examples
def make_prompt(text_to_replace):
    prompt = """
    Can you help me analyze a code-switched text in Roman Urdu and English to identify one or multiple propaganda techniques? There can also be no propaganda techniques in a text.
    The list of 20 propaganda techniques arre given below:

    - Loaded Language
    - Obfuscation, Intentional vagueness, Confusion
    - Appeal to fear/prejudice
    - Appeal to authority
    - Whataboutism
    - Slogans
    - Exaggeration/Minimisation
    - Black-and-white Fallacy/Dictatorship
    - Smears
    - Doubt
    - Bandwagon
    - Name calling/Labeling
    - Reductio ad hitlerum
    - Presenting Irrelevant Data (Red Herring)
    - Repetition
    - Misrepresentation of Someone's Position (Straw Man)
    - Thought-terminating cliché
    - Glittering generalities (Virtue)
    - Flag-waving
    - Causal Oversimplification

    The Text to analyze is: {}
    Please make sure the answer is comma seperated if multiple propaganda techniques are given.
    """.format(text_to_replace)
    return prompt


if __name__ == '__main__':

    LogFileExist = os.path.exists(os.getcwd() + '/Log_Files')
    if not LogFileExist:
        raise Exception("Log File doesn't exist")

    openai.api_key = "<PRIVATE_KEY>"

    hyper_params = {}
    hyper_params['domain_type'] = "CS"
    hyper_params["Testing_Data"] = "./Data_Files/Splits/test_split.json"

    df_test = Dataset_Preparation.read_json_files_to_df(hyper_params['Testing_Data'],hyper_params, 'Testing')

    with open('techniques.json') as file:
        techniques_pair = json.load(file)

    #########################################################################################################

    techniques_list = list(techniques_pair.keys())

    predicted_techniques, original_techniques, gold_labels_list, pred_labels_list = [], [], [], []
    for i, f in tqdm(df_test.iterrows()):
        text = f['text']
        techniques = list(set(f['technique']))

        try:
            chat_gpt_answer = get_completion(make_prompt(text), model="gpt-3.5-turbo")
            time.sleep(2)
        except:
            print('Waiting on {}'.format(i))
            time.sleep(20)
            chat_gpt_answer = get_completion(make_prompt(text), model="gpt-3.5-turbo")

        with open('chat_gpt_answers.json', "a") as json_file:
            data = {
                "id": i,
                "text": text,
                "chat_gpt": chat_gpt_answer,
            }

            json.dump(data, json_file)
            json_file.write('\n')

        splitted = re.split(r'[:,]', chat_gpt_answer)
        pred_tags_unfiltered = [item.strip() for item in splitted if item.strip() != '']
        pred_tags = [x for x in pred_tags_unfiltered if x in techniques_list]

        pred_labels  =  np.zeros((len(techniques_list)))
        for p_tag in pred_tags:
            if p_tag in techniques_pair:
                pred_labels[techniques_pair[p_tag]] = 1
            else:
                print('ERROR - {}'.format(chat_gpt_answer))
        pred_labels =  pred_labels.reshape(1,-1).astype(int)

        

        gold_indices = [techniques_pair[l] for l in techniques]
        gold_labels = np.zeros((len(techniques_list)))
        gold_labels[gold_indices] = 1
        gold_labels =  gold_labels.reshape(1,-1).astype(int)

        predicted_techniques.append(pred_tags)
        original_techniques.append(techniques)
        gold_labels_list.append(gold_labels)
        pred_labels_list.append(pred_labels)


    for i, l in enumerate(pred_labels_list):
        assert len(l) == len(gold_labels_list[i])

    #########################################################################################################
    
    gold_labels_list = np.array(gold_labels_list)
    pred_labels_list = np.array(pred_labels_list)
    gold_labels_list = np.squeeze(gold_labels_list, axis=1)
    pred_labels_list = np.squeeze(pred_labels_list, axis=1)

    labels_ = techniques_list[:]

    accuracy_score = get_accuracy_score(gold_labels_list, pred_labels_list)
    hamming_score = get_hamming_score(gold_labels_list, pred_labels_list)
    exact_match_ratio = get_exact_match_ratio(gold_labels_list, pred_labels_list)
    print('     Validation Accuracy Score: {}'.format(accuracy_score))
    print('     Validation Hamming Score: {}'.format(hamming_score))
    print('     Validation Exact Match Ratio: {}'.format(exact_match_ratio))
    classificationReport = get_classification_report(gold_labels_list, pred_labels_list, labels_)
    print('Classification Report\n')
    print(classificationReport)

# answer = get_completion(prompt, model="gpt-3.5-turbo")
# print(answer)


"""
    Here are a few examples of different code-switched texts, along with their output in the form of comma-separated propaganda techniques, provided for your guidance.

    Examples:

        Text: This movie has been directed by a gay guy iss movie ko phailainay ke bajai isko rokna chahye. They are just manipulating the whole situation. Also it won the queer award.
        Output: Reductio ad hitlerum, Smears, Loaded Language, Exaggeration/Minimisation, Appeal to fear/prejudice

        Text: Joyland is banned in Punjab but is playing in cinemas across Sindh. Achi tarah pta chal gaya kon si hakumat ghulamon ki hai.
        Output: Presenting Irrelevant Data (Red Herring), Loaded Language

        Text: We hate Babar cause all of India does. Apnay liay nai tou issiliay karlo ke hum sab bhi kartay hain
        Output: Bandwagon, Exaggeration/Minimisation, Loaded Language

        Text: As we all know in big guns circles a term used ''Get a promotion with wife swap''. Baqi aap samajhdar ho keh keya matlab hey es ka
        Output: Obfuscation, Intentional vagueness, Confusion

        Text: Aurat march is trash as rather than helping women who actually needs helps all they talk about is nanga honei do, lesbian honei dou
        Output: Misrepresentation of Someone's Position (Straw Man), Smears, Loaded Language, Name calling/Labeling, Exaggeration/Minimisation 

        Text: Either you are a Muslim woman or a feminist, koi beech ka raasta nhi, faisla kr lo and stick to it
        Output: Black-and-white Fallacy/Dictatorship

        Text: No matter how much you try, mard aur orat kabhi barabar nhi ho sakte. It is what it is
        Output: Thought-terminating cliché

        Text: Jahez ke naam pe itni takleef kyu hoti hai aurton ko, what about their demands of big house and big car from the husband?
        Output: Whataboutism, Loaded Language

        Text: I hope women get over their bad boy syndrome soon. Pehle ameer baap ki aulaad dekh ke shaadi krti hain phir domestic abuse ke randi rone, you girls know what you're getting yourself into.
        Output: Appeal to fear/prejudice, Name calling/Labeling, Loaded Language, Causal Oversimplification

        Text: Patwaari aur youthiye larte rehein ge, meanwhile Nawaz and Niyazi will loot the country and leave for England.
        Output: Name calling/Labeling, Loaded Language, Smears

        Text: If they were previously politically motivated then in saroon ka court marshall kuro. Aur saza dou saza dou saza dou saza dou.
        Output: Smears, Repetition

        Text: You are the top crypto scammer of Pakistan. Tumhain pata hai tum ne in masoom logon ke sath kya kya kiya hai kabhi ponzi scheme, kabhi courses aur kabhi paid subscriptions
        Output: Name calling/Labeling, Exaggeration/Minimisation, Smears

        Text: Death penalty is ridiculous. Aik rapist ko latka dene se baqion ko kia faraq pare ga?
        Output: Loaded Language, Doubt, Exaggeration/Minimisation

        Text: Pakistan is about to get bankrupt soon. Pack your bags and leave for Canada before it's too late
        Output: Appeal to fear/prejudice, Loaded Language

        Text: Imran Khan keh raha hai isko follow nai karna chahye tou isko unfollow karna is the best option.
        Output: Appeal to authority

        Text: Afghanion ko Pakistan se nikalo, haraamkhor humara khaa kr humein hi gaali dete hain. Also remember how they caused destruction through Kalashnikov culture in our beloved country
        Output: Flag-waving, Appeal to fear/prejudice, Name calling/Labeling

        Text: Women divorce whenever they need money. Muft mein alimony bhi lo bachon ki custody bhi
        Output: Loaded Language, Causal Oversimplification, Exaggeration/Minimisation

        Text: Utho Pakistanio! Save your country!
        Output: Slogans

        Text: Taliban are the true mujahids defending their country like lions
        Output: Glittering generalities (Virtue), Name calling/Labeling

        Text: Malik Riaz is a cult. PPP ke saath mil kar paani ke daamon zamin kharredi hai to make Bahria Town Karachi
        Output: Smears, Name calling/Labeling
"""