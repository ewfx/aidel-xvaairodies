import pandas as pd
import random

from sklearn.ensemble import IsolationForest
from transformers import pipeline
import seaborn as sns
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


mapped_columns = {
    'description': 'Transaction Details',
    'amount': 'Amount',
    'payer': 'Payer Name',
    'receiver': 'Receiver Name',
    'receiver_country': 'Receiver Country'
}

ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def load_and_enrich_data():

    ### ✅ 2. Load
    data = pd.read_csv('structured.csv')

    ### ✅ 3. Validate
    expected_columns = ['Transaction ID', 'Payer Name', 'Receiver Name', 'Transaction Details', 'Amount', 'Receiver Country']
    missing_cols = [col for col in expected_columns if col not in data.columns]
    if missing_cols:
        raise Exception(f"Missing columns: {missing_cols}")

    ### ✅ 4. Entity Extraction
    data['extracted_entities'] = data[mapped_columns['description']].apply(lambda x: extract_entities(str(x)))
    ##data.head()
    data['enrichment'] = data['extracted_entities'].apply(lambda entity_list: [enrich_entity(e) for e in entity_list])



    return data

def genRiskScoring_and_supportingEvidance(data):
    risk_features = data[[mapped_columns['amount']]].fillna(0)
    iso = IsolationForest(contamination=0.05)
    data['risk_score'] = iso.fit_predict(risk_features)

    scaling = {1: random.uniform(0.1, 0.5), -1: random.uniform(0.7, 1.0)}
    data['risk_score'] = data['risk_score'].apply(lambda x: scaling[x])
    temptable = data['enrichment']
    data['supporting_evidence'] = temptable.apply(generate_supporting_evidence)

    data['entity_type'] = data['enrichment'].apply(classify_entity)

    #data['supporting_evidence'].fillna("No College", inplace=True)



    return data




def predict_risk_score_and_entity_type(amount, receiver_country, Payer_Name,Receiver_Name,data):
    newdata = {'Amount': [amount]}
    df = pd.DataFrame(newdata)
    # Extract entities
    payer_entities = extract_entities(str(Payer_Name))
    receiver_entities = extract_entities(str(Receiver_Name))

    # Enrich entities
    payer_enrichments =  enrich_entity(payer_entities)
    receiver_enrichments = enrich_entity(receiver_entities)

    iso = IsolationForest(contamination=0.05)
    risk_features = data[[mapped_columns['amount']]].fillna(0)
    risk_score= iso.fit_predict(risk_features)

    print("payer : " + str(payer_enrichments) )
    print("receiver : " + str(receiver_enrichments))
    print('risk score : ' + str(risk_score[1]))




def search_wikidata_entity(name):
    """Search for an entity by name on Wikidata and return the best matching Q-code."""
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": name
    }
    response = requests.get(search_url, params=params)
    results = response.json().get("search", [])
    if results:
        return results[0]["id"]  # Return the best mat

##################### Utility Functions #####################

def extract_entities(text):
    entities = ner_model(text)
    return [e['word'] for e in entities if e['entity_group'] == "ORG"]

def classify_entity(enrichments):
    if any(e['watchlist_flag'] for e in enrichments):
        return 'High-Risk Entity'
    elif any(e['registered_country'] in ['Cayman Islands', 'Panama'] for e in enrichments):
        return 'Shell Company'
    else:
        return random.choice(['Corporation', 'Non-Profit', 'Government Agency'])


def generate_supporting_evidence(enrichments):
    return "; ".join([f"Sector: {e['business_sector']}, Country: {e['registered_country']}, Watchlist: {e['watchlist_flag']}" for e in enrichments])

def enrich_entity(entity_name):
    return {
        'business_sector': random.choice(['Finance', 'Technology', 'Healthcare', 'Construction', 'Non-Profit']),
        'registered_country': random.choice(['US', 'UK', 'Cayman Islands', 'India', 'Panama']),
        'watchlist_flag': random.choice([True, False])
    }