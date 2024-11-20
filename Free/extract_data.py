import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Initialize Firebase
cred = credentials.Certificate('midlv1-firebase-adminsdk-82s34-7972eabb7b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Reference to the 'jobs' collection
collection_ref = db.collection('jobs')

# Retrieve documents from the collection
docs = collection_ref.stream()

# Extract data from documents
data = []
for doc in docs:
    job_data = doc.to_dict()
    created_by_ref = job_data.get('created_by')
    if created_by_ref:
        user_doc_id = created_by_ref.id  # Extract the document ID
        user_doc = db.document(created_by_ref.path).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            user_values = user_data.get('values', [])  # Get the 'values' list from user data or provide an empty list if it doesn't exist
            job_data['created_by'] = {'uid': user_doc_id, 'values': user_values}  # Ensure 'values' is included in the 'created_by' dictionary
        else:
            job_data['created_by'] = {'uid': user_doc_id, 'values': []}  # Provide an empty list if user doc does not exist
    data.append(job_data)

# Create DataFrame from the extracted data
df = pd.DataFrame(data)

# Split DataFrame into company and candidate jobs
company_df = df[df['job_type'] == 'entreprise']
candidate_df = df[df['job_type'] == 'candidat']

# Export DataFrames to CSV files
company_df.to_csv('companies_job.csv', index=False)
candidate_df.to_csv('candidates_job.csv', index=False)

print('The data has been exported to companies_jobs.csv and candidates_jobs.csv')
