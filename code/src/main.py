from tensorflow.python.ops.metrics_impl import false_negatives

import train

def main():

    #LOAD DATA
    data = train.load_and_enrich_data()

    #ENRICH DATA
    enrichedData = train.genRiskScoring_and_supportingEvidance(data)

    #PREDICT OUTPUT
    train.predict_risk_score_and_entity_type(5000, 'China', 'Global Ventures', 'Innovate Systems',enrichedData)




if __name__ == "__main__":
    main()