import openai

API_KEY = "sk-proj-dSfBlLvnwQN52jwW5dmQT3BlbkFJXCqK1Dn2g5WdBc8ruZl4"
openai.api_key = API_KEY

def generate_interview_questions(profile_description, cv_text, offer_description):
    import openai
    
    messages = [
        {
            "role": "system",
            "content": "Vous êtes un expert en ressources humaines spécialisé dans la conduite d'entretiens d'embauche."
        },
        {
            "role": "user",
            "content": f"""
            Je vais vous fournir une description du profil recherché par l'entreprise, le CV d'un candidat et une description de l'offre d'emploi. Générer 4 questions d'entretien pour ce candidat : deux questions spécifiques à l'offre d'emploi, et deux questions spécifiques au CV du candidat.

            Description du profil recherché :
            {profile_description}

            CV du candidat :
            {cv_text}

            Description de l'offre :
            {offer_description}

            Questions d'entretien :
            """
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150,
        n=1,
        temperature=0.5,
    )
    
    generated_questions = response['choices'][0]['message']['content'].strip()
    
    # Add the two initial questions
    initial_questions = [
        "Parlez-moi de vous?",
        "Quelles sont vos forces et vos faiblesses?"
    ]
    
    # Combine the initial questions with the generated questions
    all_questions = initial_questions + generated_questions.split('\n')
    
    # Strip any leading or trailing whitespace from each question
    formatted_questions = [question.strip() for question in all_questions if question.strip()]
    
    return formatted_questions



# Profile Description (what the company is looking for)
profile_description = """
TechCorp recherche un développeur Full Stack passionné par les nouvelles technologies et l'innovation. Le candidat idéal aura une solide expérience en développement web et mobile, une aptitude à travailler en équipe, et une capacité à résoudre des problèmes complexes. Il doit être proactif, capable de gérer plusieurs projets simultanément et avoir une forte motivation pour l'apprentissage continu. La maîtrise de JavaScript, React, et Node.js est essentielle, ainsi qu'une bonne connaissance des bases de données SQL et NoSQL.
"""

# User's CV
cv_text = """
Nom : Jean Dupont
Adresse : 123 Rue de l'Université, 75005 Paris, France
Téléphone : +33 6 12 34 56 78
Email : jean.dupont@example.com

**Résumé :**
Développeur logiciel passionné avec plus de 3 ans d'expérience dans le développement web et mobile. Compétences solides en JavaScript, React, Node.js et Python. Expérience avérée dans la gestion de projets et le travail en équipe.

**Expérience professionnelle :**
- **TechCorp, Paris, France** (Jan 2021 - Présent)
  - Développeur Web
  - Responsabilités : Développement et maintenance de sites web en utilisant JavaScript, React et Node.js. Collaboration avec les équipes de conception et de marketing pour améliorer l'expérience utilisateur.

- **Innovatech, Paris, France** (Sept 2018 - Déc 2020)
  - Développeur Mobile
  - Responsabilités : Développement d'applications mobiles pour Android et iOS en utilisant React Native. Participation à des revues de code et à des tests unitaires pour assurer la qualité du code.

**Formation :**
- **Université de Paris, France** (2015 - 2018)
  - Licence en Informatique

**Compétences :**
- Langages de programmation : JavaScript, Python, Java, C++
- Technologies web : HTML, CSS, React, Node.js
- Outils de développement : Git, Docker, Jenkins
- Langues : Français (natif), Anglais (courant)

**Projets personnels :**
- **OpenSource Contributor** : Contribution à divers projets open source sur GitHub, notamment dans le domaine de l'IA et du développement web.

**Centres d'intérêt :**
- Participation à des hackathons
- Lecture de livres techniques
- Randonnée et voyages
"""

# Offer Description
offer_description = """
**Titre du poste :** Développeur Full Stack

**Entreprise :** TechCorp

**Lieu :** Paris, France

**Description de l'offre :**
TechCorp recherche un Développeur Full Stack talentueux et motivé pour rejoindre notre équipe dynamique. Vous serez responsable du développement et de la maintenance de nos applications web et mobiles. Vous travaillerez en étroite collaboration avec les équipes de conception, de produit et de marketing pour créer des expériences utilisateur exceptionnelles.

**Responsabilités :**
- Développer des fonctionnalités front-end en utilisant React
- Créer et maintenir des API back-end avec Node.js
- Collaborer avec les équipes de conception pour traduire les maquettes en fonctionnalités
- Participer à des revues de code et assurer la qualité du code
- Contribuer à l'amélioration continue de nos processus de développement

**Qualifications requises :**
- Licence en Informatique ou domaine connexe
- Minimum de 2 ans d'expérience en développement full stack
- Maîtrise de JavaScript, React, Node.js
- Connaissance des bases de données SQL et NoSQL
- Expérience avec Git et les outils CI/CD
- Excellentes compétences en communication et en travail d'équipe

**Pourquoi rejoindre TechCorp :**
- Environnement de travail innovant et collaboratif
- Opportunités de développement professionnel et de formation continue
- Avantages sociaux compétitifs et salaire attractif
- Culture d'entreprise centrée sur l'innovation et l'excellence
"""

# Generate interview questions
questions = generate_interview_questions(profile_description, cv_text, offer_description)
print("Generated Interview Questions:")
for question in questions:
    print(question)
