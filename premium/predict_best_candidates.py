import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from openai import OpenAI


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

# Print column names for debugging
print("Offers DataFrame columns:", offers_df.columns)
print("Candidates DataFrame columns:", candidates_df.columns)

# Ensure all values in relevant columns are strings
relevant_columns = [
    'technical_skills', 'remote_status', 'job_name', 'salary', 'job_time', 
    'distance', 'contract_time', 'experience', 'availability', 'advantages', 
    'contract_types', 'study_level', 'soft_skills'
]
for col in relevant_columns:
    offers_df[col] = offers_df[col].astype(str)
    candidates_df[col] = candidates_df[col].astype(str)

# Fill missing values in 'technical_skills' and 'soft_skills' with empty strings
candidates_df['technical_skills'] = candidates_df['technical_skills'].fillna('')
offers_df['technical_skills'] = offers_df['technical_skills'].fillna('')
candidates_df['soft_skills'] = candidates_df['soft_skills'].fillna('')
offers_df['soft_skills'] = offers_df['soft_skills'].fillna('')

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
soft_skills_mapping = {
    'Organisation 📁': 1,
    'Résilience 💪': 2,
    'Responsabilité ✅': 3,
    'Ponctualité ⏰': 4,
    'Innovation 🆕': 5,
    'Analyse 📊': 6,
    'Proactivité 🌟': 7,
    'Éthique ✨': 8,
    'Créativité 🎨': 9,
    'Esprit d\'équipe 👥': 10,
    'Autonomie 🏹': 11,
    'Gestion du temps ⏳': 12,
    'Motivation 🔥': 13,
    'Écoute 👂': 14,
    'Optimisme ☀️': 15,
    'Tolérance 🌈': 16,
    'Flexibilité 🌠': 17,
    'Apprentissage 📚': 18,
    'Gestion de conflits 🕊️': 19,
    'Prise de décision 🤔': 20,
    'Collaboration 🤝': 21,
    'Gestion des priorités 📝': 22,
    'Sens critique 🔍': 23,
    'Intelligence émotionnelle 💓': 24,
    'Vision stratégique 🌍': 25,
    'Diplomatie 🕊️': 26,
    'Curiosité intellectuelle 🤓': 27,
    'Adaptation rapide ⚡': 28,
    'Gestion de projet 📈': 29,
    'Influence positive 🌞': 30,
    'Communication 🌐': 31,
    'Négociation 🤝': 32,
    'Écoute active 👂': 33,
    'Confidentialité 🤫': 34,
    'Passionné ❤️': 35,
    'Enthousiasme 😄': 36,
    'Persévérance 🏋️': 37,
    'Empathie 🤗': 38,
    'Adaptabilité 🔄': 39,
    'Initiative 💡': 40,
    'Gestion du stress 🧘': 41,
    'Leadership 👑': 42
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
# Mapping pour les villes 
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



# Encode remote status, availability, contract types, job time, experience, education, salary, soft skills, and values for candidates
candidates_df['remote_status_encoded'] = candidates_df['remote_status'].map(remote_status_mapping)
candidates_df['availability_encoded'] = candidates_df['availability'].map(availability_mapping)
candidates_df['contract_type_encoded'] = candidates_df['contract_types'].map(contract_type_mapping)
candidates_df['job_time_encoded'] = candidates_df['job_time'].map(job_time_mapping)
candidates_df['experience_encoded'] = candidates_df['experience'].map(experience_mapping)
candidates_df['education_encoded'] = candidates_df['study_level'].map(education_mapping)
candidates_df['salary_encoded'] = candidates_df['salary'].map(salary_mapping)
candidates_df['localisation_num'] = candidates_df['address'].map(location_mapping)
# Split soft skills and values, and apply mapping
candidates_df['soft_skills_encoded'] = candidates_df['soft_skills'].apply(lambda x: [soft_skills_mapping[skill.strip()] for skill in x.split('; ') if skill.strip() in soft_skills_mapping])
candidates_df['values_encoded'] = candidates_df['values'].apply(lambda x: [values_mapping[value.strip()] for value in x if value.strip() in values_mapping])

# Fill NaN values with defaults
candidates_df.fillna({
    'remote_status_encoded': 3,  # Default to 'Indifférent'
    'availability_encoded': 4,  # Default to '📅 + de 3 mois'
    'contract_type_encoded': 0,  # Default to '📃 CDI'
    'job_time_encoded': 0,  # Default to '⏰ Temps plein'
    'experience_encoded': 0,  # Default to '🌱 Sans expérience'
    'education_encoded': 0,  # Default to '📖 Autodidacte'
    'salary_encoded': 0  # Default to '💰 Jusqu’à 20 000 €'
}, inplace=True)

# Repeat encoding for offers
offers_df['remote_status_encoded'] = offers_df['remote_status'].map(remote_status_mapping)
offers_df['availability_encoded'] = offers_df['availability'].map(availability_mapping)
offers_df['contract_type_encoded'] = offers_df['contract_types'].map(contract_type_mapping)
offers_df['job_time_encoded'] = offers_df['job_time'].map(job_time_mapping)
offers_df['experience_encoded'] = offers_df['experience'].map(experience_mapping)
offers_df['education_encoded'] = offers_df['study_level'].map(education_mapping)
offers_df['salary_encoded'] = offers_df['salary'].map(salary_mapping)
offers_df['soft_skills_encoded'] = offers_df['soft_skills'].apply(lambda x: [soft_skills_mapping[skill.strip()] for skill in x.split('; ') if skill.strip() in soft_skills_mapping])
offers_df['values_encoded'] = offers_df['values'].apply(lambda x: [values_mapping[value.strip()] for value in x if value.strip() in values_mapping])

# Fill NaN values with defaults for offers
offers_df.fillna({
    'remote_status_encoded': 3,  # Default to 'Indifférent'
    'availability_encoded': 4,  # Default to '📅 + de 3 mois'
    'contract_type_encoded': 0,  # Default to '📃 CDI'
    'job_time_encoded': 0,  # Default to '⏰ Temps plein'
    'experience_encoded': 0,  # Default to '🌱 Sans expérience'
    'education_encoded': 0,  # Default to '📖 Autodidacte'
    'salary_encoded': 0  # Default to '💰 Jusqu’à 20 000 €'
}, inplace=True)

# Convert lists of soft skills and values to binary matrices
def encode_skills_and_values(skill_lists, value_lists, all_skills, all_values):
    """
    Encode lists of soft skills and values into binary matrices.

    Args:
        skill_lists (list): List of soft skills for each candidate/offer.
        value_lists (list): List of values for each candidate/offer.
        all_skills (list): List of all possible soft skills.
        all_values (list): List of all possible values.

    Returns:
        tuple: A tuple containing the encoded soft skills matrix and values matrix.
    """
    skills_matrix = []
    values_matrix = []
    for skills, values in zip(skill_lists, value_lists):
        skills_row = [1 if skill in skills else 0 for skill in all_skills]
        values_row = [1 if value in values else 0 for value in all_values]
        skills_matrix.append(skills_row)
        values_matrix.append(values_row)
    return pd.DataFrame(skills_matrix, columns=all_skills), pd.DataFrame(values_matrix, columns=all_values)

# Get lists of all possible soft skills and values
all_soft_skills = list(soft_skills_mapping.values())
all_values = list(values_mapping.values())

# Encode soft skills and values for candidates and offers
candidates_skills_matrix, candidates_values_matrix = encode_skills_and_values(candidates_df['soft_skills_encoded'], candidates_df['values_encoded'], all_soft_skills, all_values)
offers_skills_matrix, offers_values_matrix = encode_skills_and_values(offers_df['soft_skills_encoded'], offers_df['values_encoded'], all_soft_skills, all_values)

# TF-IDF Vectorization for technical skills
vectorizer = TfidfVectorizer()
candidates_tech_skills_tfidf = vectorizer.fit_transform(candidates_df['technical_skills'])
offers_tech_skills_tfidf = vectorizer.transform(offers_df['technical_skills'])

# Combine all features into a single DataFrame for candidates and offers
candidates_features = pd.concat([
    candidates_df[['remote_status_encoded', 'availability_encoded', 'contract_type_encoded', 'job_time_encoded', 'experience_encoded', 'education_encoded', 'salary_encoded']],
    pd.DataFrame(candidates_tech_skills_tfidf.toarray(), index=candidates_df.index, columns=vectorizer.get_feature_names_out()),
    candidates_skills_matrix,
    candidates_values_matrix
], axis=1)

offers_features = pd.concat([
    offers_df[['remote_status_encoded', 'availability_encoded', 'contract_type_encoded', 'job_time_encoded', 'experience_encoded', 'education_encoded', 'salary_encoded']],
    pd.DataFrame(offers_tech_skills_tfidf.toarray(), index=offers_df.index, columns=vectorizer.get_feature_names_out()),
    offers_skills_matrix,
    offers_values_matrix
], axis=1)

# Ensure the features have the same columns for both candidates and offers
common_columns = candidates_features.columns.intersection(offers_features.columns)
candidates_features = candidates_features[common_columns]
offers_features = offers_features[common_columns]

# Calculate cosine similarity between candidates and offers
similarity_matrix = cosine_similarity(candidates_features, offers_features)

def get_all_best_match_offers(candidate_id):
    """
    Get the top 10 best match job offers for a given candidate based on cosine similarity.

    Args:
        candidate_id (str): The unique identifier of the candidate.

    Returns:
        list: A list of dictionaries containing the best match job offers, their similarity scores, and personality comparisons.
    """
    candidate_index = candidates_df[candidates_df['uid'] == candidate_id].index
    if len(candidate_index) == 0:
        return "Candidate not found"
    candidate_index = candidate_index[0]
    best_match_indices = similarity_matrix[candidate_index].argsort()[::-1][:10]
    best_matches = []
    candidate_profile = candidates_df.iloc[candidate_index]['soft_skills']

    for idx in best_match_indices:
        best_match_job = offers_df.iloc[idx]['job_name']
        best_match_company = offers_df.iloc[idx]['uid']
        similarity_score = similarity_matrix[candidate_index][idx]

        # Comparaison des personnalités entre le candidat et l'offre
        recruiter_profile = offers_df.iloc[idx]['soft_skills']
        personality_comparison = comparer_personnalites(candidate_profile, recruiter_profile)

        best_matches.append({
            'Best Match Job': best_match_job,
            'Best Match Company': best_match_company,
            'Similarity Score': similarity_score,
            'Personality Comparison': personality_comparison
        })
    return best_matches



client = OpenAI(api_key="sk-proj-dSfBlLvnwQN52jwW5dmQT3BlbkFJXCqK1Dn2g5WdBc8ruZl4")
def comparer_personnalites(soft_skills_candidate, soft_skills_recruiter):
    """
    Compare les personnalités du candidat et du recruteur en utilisant le modèle GPT-4 d'OpenAI.

    Args:
        candidate_profile (str): Profil du candidat.
        recruiter_profile (str): Profil du recruteur.

    Returns:
        str: Résultat de la comparaison des personnalités.
    """
    client = OpenAI(api_key="sk-proj-dSfBlLvnwQN52jwW5dmQT3BlbkFJXCqK1Dn2g5WdBc8ruZl4")
    prompt = f"""
    Comparez les personnalités suivantes et expliquez les différences et les similarités en termes de traits de personnalité, styles de communication et compatibilité professionnelle.

    Profil du candidat :
    {soft_skills_candidate}

    Profil du recruteur :
    {soft_skills_recruiter}

    Veuillez structurer votre comparaison comme suit :
    - Traits de personnalité : (liste des traits de personnalité du candidat et du recruteur)
    - Styles de communication : (analyse des styles de communication du candidat et du recruteur)
    - Compatibilité professionnelle : (évaluation de la compatibilité professionnelle entre le candidat et le recruteur)
    """

    response = client.chat.completions.create(
        model="gpt-4",  # Utilisez le modèle approprié
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content.strip()



# Exemple d'utilisation
candidate_id = 'hVI9m1QnViUgvCNkyoP9wGosuDU2'  # Remplacez par l'ID du candidat que vous souhaitez analyser
results = get_all_best_match_offers(candidate_id)
for result in results:
    print("Offre d'emploi:", result['Best Match Job'])
    print("Entreprise:", result['Best Match Company'])
    print("Score de similarité:", result['Similarity Score'])
    print("Comparaison de personnalités:")
    print(result['Personality Comparison'])

