import pandas as pd
from tqdm import tqdm
from google.cloud import translate


def get_language_codes():
    parent = f"projects/roman-urdu-translation-api"
    client = translate.TranslationServiceClient()

    response = client.get_supported_languages(parent=parent, display_language_code="en")
    languages = response.languages

    print(f" Languages: {len(languages)} ".center(60, "-"))
    for language in languages:
        print(f"{language.language_code}\t{language.display_name}")

def translate_text(text):

    parent = f"projects/roman-urdu-translation-api"
    client = translate.TranslationServiceClient()

    
    # Romanian to Urdu
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "ro",
            "target_language_code": "ur",
        }
    )

    for translation in response.translations:
        # print("Translated text: {}".format(translation.translated_text))
        text = translation.translated_text

    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "ur",
            "target_language_code": "en-US",
        }
    )

    for translation in response.translations:
        # print("Translated text: {}".format(translation.translated_text))
        final_text = translation.translated_text

    return final_text


def apply_translate(df):
    df['TRANSLATED'] = ""
    for i, f in tqdm(df.iterrows()):
        try:
            tr_text = translate_text(f['TEXT'])
            df.at[i, 'TRANSLATED'] = tr_text
        except Exception as e:
            print(i)
            print(e)
    
    return df



if __name__ == '__main__':

    df = pd.read_csv('Dataset.csv')
    final_df = apply_translate(df)
    final_df.to_csv('Translated_dataset.csv' ,index=False)
    

    # example_text = "kuch bhi nai ho raha"
    # translated_text = translate_text(example_text)
    # print(translated_text)
    # get_language_codes()


##################### SET UP API #####################
#-----------------------------------------------------
# Link to help: https://codelabs.developers.google.com/codelabs/cloud-translation-python3#3
# Create a Project
# Project ID = roman-urdu-translation-api (Can get this at the start while creating the project or running "gcloud config get-value core/project" in the cloud shell)
# Go to API & Services
# Search for "Cloud Translation API"            # List of API's https://cloud.google.com/translate/docs/reference/rest/?apix=true
# Enable API
# Go to Credentials
# Don't use this (APi Keys)
######################
# API_KEY = 
# Use this key in your application by passing it with the key=API_KEY parameter.
######################
# For server to server through app we need Service Accounts
# Create a service account
# After creating service account go to keys
# Create a new JSON key
# On your server
# export GOOGLE_APPLICATION_CREDENTIALS=/home/umar/Desktop/Thesis/Propaganda_Detection_on_Code_Switched_Data/Dataset/roman-urdu-translation-api-84ae93a0ad5e.json

# Each time you wish to run you will have to export GOOGLE_APPLICATION_CREDENTIALS