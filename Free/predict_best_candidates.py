import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

# Load datasets
offers_df = pd.read_csv('companies_job.csv')
candidates_df = pd.read_csv("candidates_job.csv")

def parse_created_by(field):
    """
    Parse the 'created_by' field from the dataset to extract user ID and values.

    Args:
        field (str): The 'created_by' field in string representation of a dictionary.

    Returns:
        tuple: A tuple containing the user ID (uid) and a list of values.
    """
    parsed = ast.literal_eval(field)
    uid = parsed['uid']
    values = parsed.get('values', [])
    return uid, values

# Apply the parsing function to extract 'uid' and 'values' into separate columns
offers_df[['uid', 'values']] = offers_df['created_by'].apply(lambda x: pd.Series(parse_created_by(x)))
candidates_df[['uid', 'values']] = candidates_df['created_by'].apply(lambda x: pd.Series(parse_created_by(x)))

# Ensure all values in relevant columns are strings
relevant_columns = ['remote_status', 'job_name', 'salary', 'job_time', 
                    'distance', 'contract_time', 'experience', 'availability', 
                    'advantages', 'contract_types', 'study_level']
for col in relevant_columns:
    offers_df[col] = offers_df[col].astype(str)
    candidates_df[col] = candidates_df[col].astype(str)

# Define mappings for additional attributes
remote_status_mapping = {
    'Non': 0,
    'Oui': 1,
    'Full Remote': 2,
    'Indifférent': 3
}
availability_mapping = {
    '📅 Immédiate': 0,
    '📅 1 mois max': 1,
    '📅 2 mois max': 2,
    '📅 3 mois max': 3,
    '📅 + de 3 mois': 4
}
contract_type_mapping = {
    '📃 CDI': 0,
    '📃 CDD': 1,
    '📃 Freelance': 2,
    '📃 Internship': 3,
    '📃 Part-time': 4
}
job_time_mapping = {
    '⏰ Temps plein': 0,
    '⏰ Temps partiel': 1
}
experience_mapping = {
    '🌱 Sans expérience': 0,
    '🌿 1 à 3 ans d\'expérience': 1,
    '🌳 3 à 5 ans d\'expérience': 2,
    '🌲 5 à 10 ans d\'expérience': 3,
    '🏞️ Plus de 10 ans d\'expérience': 4
}
education_mapping = {
    '📖 Autodidacte': 0,
    '📜 Diplôme national du brevet': 1,
    '🛠️ CAP': 2,
    '📚 BEP': 3,
    '🎓 Bac': 4,
    '🏫 DUT': 5,
    '🧑‍🔬 BTS': 6,
    '🎓 📚 Licence': 7,
    '🎓 📚 📚 Master': 8,
    '🎓 🔬 Doctorat': 9
}
salary_mapping = {
    '💰 Jusqu’à 20 000 €': 0,
    '💰 De 20 000 à 30 000 €': 1,
    '💰 De 30 000 à 40 000 €': 2,
    '💰 De 40 000 à 55 000 €': 3,
    '💰 De 55 000 à 70 000 €': 4,
    '💰 ➕ de 70.000 €': 5
}

values_mapping = {
    'Égalité des chances 🌈': 1,
    'Diversité et inclusion 🌍': 2,
    'NoRacism 🛑': 3,
    'Droits de l\'homme ⚖️': 4,
    'Égalité des sexes': 5,
    'Protection de l\'environnement 🌿': 6,
    'Durabilité et écoresponsabilité 🌱': 7,
    'Justice sociale 🎓': 8,
    'Soutien à l\'éducation 📚': 9,
    'Lutte contre la pauvreté ✊': 10,
    'Accessibilité pour tous ♿': 11,
    'Aide humanitaire 🏥': 12,
    'Développement de compétences 🛠️': 13,
    'Respect de la vie privée et des données 🔐': 14,
    'Action pour le climat ☀️': 15
}

location_mapping = {
    "Paris": 1,
    "Marseille": 2,
    "Lyon": 3,
    "Toulouse": 4,
    "Nice": 5,
    "Nantes": 6,
    "Strasbourg": 7,
    "Montpellier": 8,
    "Bordeaux": 9,
    "Lille": 10,
    "Rennes": 11,
    "Reims": 12,
    "Le Havre": 13,
    "Cergy": 14,
    "Saint-Étienne": 15,
    "Toulon": 16,
    "Angers": 17,
    "Grenoble": 18,
    "Dijon": 19,
    "Nîmes": 20,
    "Aix-en-Provence": 21,
    "Saint-Quentin-en-Yvelines": 22,
    "Brest": 23,
    "Le Mans": 24,
    "Amiens": 25,
    "Limoges": 26,
    "Tours": 27,
    "Clermont-Ferrand": 28,
    "Villeurbanne": 29,
    "Metz": 30,
    "Besançon": 31,
    "Orléans": 32,
    "Mulhouse": 33,
    "Caen": 34,
    "Rouen": 35,
    "Boulogne-Billancourt": 36,
    "Perpignan": 37,
    "Nancy": 38,
    "Roubaix": 39,
    "Argenteuil": 40,
    "Montreuil": 41,
    "Avignon": 42,
    "Versailles": 43,
    "Pau": 44,
    "La Rochelle": 45,
    "Calais": 46,
    "Antibes": 47,
    "Saint-Denis": 48,
    "Saint-Paul (La Réunion)": 49,
    "Le Tampon (La Réunion)": 50
}

# Encode categorical features
candidates_df['remote_status_encoded'] = candidates_df['remote_status'].map(remote_status_mapping)
candidates_df['availability_encoded'] = candidates_df['availability'].map(availability_mapping)
candidates_df['contract_type_encoded'] = candidates_df['contract_types'].map(contract_type_mapping)
candidates_df['job_time_encoded'] = candidates_df['job_time'].map(job_time_mapping)
candidates_df['experience_encoded'] = candidates_df['experience'].map(experience_mapping)
candidates_df['education_encoded'] = candidates_df['study_level'].map(education_mapping)
candidates_df['salary_encoded'] = candidates_df['salary'].map(salary_mapping)
candidates_df['localisation_num'] = candidates_df['address'].map(location_mapping)

offers_df['remote_status_encoded'] = offers_df['remote_status'].map(remote_status_mapping)
offers_df['availability_encoded'] = offers_df['availability'].map(availability_mapping)
offers_df['contract_type_encoded'] = offers_df['contract_types'].map(contract_type_mapping)
offers_df['job_time_encoded'] = offers_df['job_time'].map(job_time_mapping)
offers_df['experience_encoded'] = offers_df['experience'].map(experience_mapping)
offers_df['education_encoded'] = offers_df['study_level'].map(education_mapping)
offers_df['salary_encoded'] = offers_df['salary'].map(salary_mapping)

# Fill NaN values with defaults
candidates_df.fillna({
    'remote_status_encoded': 3,  # Default to 'Indifférent'
    'availability_encoded': 4,   # Default to '📅 + de 3 mois'
    'contract_type_encoded': 0,  # Default to '📃 CDI'
    'job_time_encoded': 0,       # Default to '⏰ Temps plein'
    'experience_encoded': 0,     # Default to '🌱 Sans expérience'
    'education_encoded': 0,      # Default to '📖 Autodidacte'
    'salary_encoded': 0           # Default to '💰 Jusqu’à 20 000 €'
}, inplace=True)

offers_df.fillna({
    'remote_status_encoded': 3,
    'availability_encoded': 4,   # Default to '📅 + de 3 mois'
    'contract_type_encoded': 0,  # Default to '📃 CDI'
    'job_time_encoded': 0,       # Default to '⏰ Temps plein'
    'experience_encoded': 0,     # Default to '🌱 Sans expérience'
    'education_encoded': 0,      # Default to '📖 Autodidacte'
    'salary_encoded': 0          # Default to '💰 Jusqu’à 20 000 €'
}, inplace=True)

# Drop unnecessary columns after encoding
candidates_df.drop(['remote_status', 'availability', 'contract_types', 'job_time', 'experience', 'study_level', 'salary'], axis=1, inplace=True)
offers_df.drop(['remote_status', 'availability', 'contract_types', 'job_time', 'experience', 'study_level', 'salary'], axis=1, inplace=True)

# Calculate cosine similarity between candidates and offers based on encoded features
candidates_features = candidates_df[['remote_status_encoded', 'availability_encoded', 'contract_type_encoded',
                                     'job_time_encoded', 'experience_encoded', 'education_encoded', 'salary_encoded']]
offers_features = offers_df[['remote_status_encoded', 'availability_encoded', 'contract_type_encoded',
                             'job_time_encoded', 'experience_encoded', 'education_encoded', 'salary_encoded']]

similarity_matrix = cosine_similarity(candidates_features, offers_features)

def get_all_best_match_candidates(offer_id):
    """
    Get the top 10 best match candidates for a given job offer based on cosine similarity.

    Args:
        offer_id (str): The unique identifier of the job offer.

    Returns:
        list: A list of dictionaries containing the best match candidates and their similarity scores.
    """
    offer_index = offers_df[offers_df['uid'] == offer_id].index
    if len(offer_index) == 0:
        return "Offer not found"
    offer_index = offer_index[0]
    best_match_indices = similarity_matrix[:, offer_index].argsort()[::-1][:10]
    best_matches = []
    for idx in best_match_indices:
        best_match_candidate = candidates_df.iloc[idx]['uid']
        similarity_score = similarity_matrix[idx, offer_index]
        best_matches.append({
            'Best Match Candidate': best_match_candidate,
            'Similarity Score': similarity_score
        })
    return best_matches

# Example usage
offer_id = 'DLM61VJ6lJgrrE14AORpNmJVrfi1'  # Change this to the ID of the job offer you want to find the best match candidates for
results = get_all_best_match_candidates(offer_id)
for result in results:
    print(result)
