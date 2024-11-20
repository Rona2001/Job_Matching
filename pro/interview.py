import whisper
from moviepy.editor import VideoFileClip
import openai
from generate_question import generate_interview_questions



API_Key = "sk-proj-dSfBlLvnwQN52jwW5dmQT3BlbkFJXCqK1Dn2g5WdBc8ruZl4"
openai.api_key = API_Key  # Set the API key directly

# Load the Whisper model
model = whisper.load_model("base")
model.device

def extract_audio(input_video):
    video = VideoFileClip(input_video)
    audio = video.audio
    output_audio = "audio.mp3"
    audio.write_audiofile(output_audio)
    video.close()
    audio.close()
    return output_audio

def transcribe(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def process_videos(video_files):
    transcriptions = []
    for video_file in video_files:
        audio_path = extract_audio(video_file)
        transcription = transcribe(audio_path)
        transcriptions.append(transcription)
    return transcriptions

def get_chatgpt_feedback(profile_description, cv_text, offer_description, question, answer):
    prompt = f"""
    Vous êtes un expert en coaching d'entretien. Je vais vous fournir une description de profil d'un candidat, son CV, une description de l'offre, une question et leur réponse.
    Fournissez des commentaires détaillés sur leur réponse et proposez des améliorations.

    Description du profil :
    {profile_description}

    CV du candidat :
    {cv_text}

    Description de l'offre :
    {offer_description}

    Question d'entretien :
    {question}

    Réponse du candidat :
    {answer}

    Commentaires :
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message['content'].strip()

def get_overall_feedback(feedbacks):
    prompt = f"""
    Vous êtes un expert en coaching d'entretien. Je vais vous fournir des commentaires détaillés sur cinq questions d'entretien différentes pour un candidat.
    Fournissez des conseils globaux et des suggestions d'amélioration basés sur ces commentaires.

    Commentaires sur les questions d'entretien :
    {"".join(feedbacks)}

    Conseils globaux :
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert analyst."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message['content'].strip()


# List of video files
video_files = ["question1.mp4", "question2.mp4", "question3.mp4", "question4.mp4", "question5.mp4"]

# Process the videos and get the transcriptions
# transcriptions = process_videos(video_files)
transcriptions = [
    "Je m'appelle Jean Dupont, et je suis diplômé en informatique de l'Université de Paris. Pendant mes études, j'ai développé une passion pour le développement logiciel et la résolution de problèmes complexes. J'ai effectué un stage chez TechCorp où j'ai travaillé sur des projets de développement web en utilisant des technologies telles que JavaScript, React, et Node.js. En dehors du travail, j'aime participer à des hackathons et contribuer à des projets open source. Je suis très motivé par l'apprentissage continu et la collaboration avec des équipes diversifiées.",
    "Je souhaite ce poste parce qu'il représente une opportunité parfaite pour moi de mettre en pratique mes compétences en développement logiciel dans un environnement innovant. Votre entreprise est reconnue pour son approche avant-gardiste et ses projets ambitieux, ce qui correspond parfaitement à mes aspirations professionnelles. De plus, je suis particulièrement attiré par les valeurs de votre entreprise et la possibilité de travailler sur des projets qui ont un impact significatif sur les utilisateurs finaux.",
    "Parmi mes forces, je peux citer ma capacité à résoudre des problèmes de manière créative et mon aptitude à travailler efficacement en équipe. Je suis également très organisé et capable de gérer plusieurs tâches simultanément sans compromettre la qualité du travail. En ce qui concerne mes faiblesses, je dirais que j'ai tendance à être perfectionniste, ce qui peut parfois ralentir mon rythme de travail. Cependant, j'ai appris à mieux gérer cela en fixant des délais réalistes et en priorisant les tâches les plus importantes.",
    "Lors de mon stage chez TechCorp, j'ai été confronté à un projet où nous devions intégrer une nouvelle API dans un délai très court. L'API était complexe et la documentation insuffisante. Pour surmonter ce défi, j'ai d'abord organisé une réunion avec l'équipe pour discuter des meilleures stratégies d'intégration. Ensuite, j'ai pris l'initiative de contacter le support technique de l'API pour obtenir des éclaircissements. En collaborant étroitement avec mes collègues et en effectuant des tests rigoureux, nous avons réussi à intégrer l'API à temps et sans bugs majeurs.",
    "Dans cinq ans, je me vois avoir évolué dans mon rôle actuel et avoir acquis une expertise significative dans le développement de logiciels innovants. J'espère également avoir eu l'occasion de prendre des responsabilités de gestion de projet et de mentorat pour aider les nouveaux membres de l'équipe à se développer. Je suis passionné par l'idée de continuer à apprendre et à m'adapter aux nouvelles technologies, et j'aimerais jouer un rôle clé dans les projets de votre entreprise qui façonnent l'avenir du secteur technologique."
]

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

# Profile Description
profile_description = """
Jean Dupont est un développeur logiciel passionné par l'innovation technologique et la résolution de problèmes complexes. Avec une solide formation académique et plusieurs années d'expérience professionnelle, Jean a acquis des compétences approfondies en développement web et mobile. Il est reconnu pour sa capacité à travailler efficacement en équipe et à mener à bien des projets de développement de bout en bout. En dehors du travail, Jean aime participer à des hackathons et contribuer à des projets open source, ce qui témoigne de son engagement continu envers l'apprentissage et l'amélioration de ses compétences techniques.
"""


questions = generate_interview_questions(profile_description, cv_text, offer_description)
print("Generated Interview Questions:")
for question in questions:
    print(question)


# Get feedback for each question
feedback = []
for i, transcription in enumerate(transcriptions):
    feedback_text = get_chatgpt_feedback(profile_description, cv_text, offer_description, questions[i], transcription)
    feedback.append(feedback_text)
    print(f"Commentaires pour la question {i+1} :")
    print(feedback_text)
    print()

# Get overall feedback
overall_feedback = get_overall_feedback(feedback)
print("Conseils globaux :")
print(overall_feedback)
