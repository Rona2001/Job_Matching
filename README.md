# Job Matching

This project leverages **Machine Learning** techniques to efficiently match candidate profiles with job opportunities.

---

## Dataset Description

This project utilizes two CSV files extracted from a Firebase jobs collection:

- **`companies_job.csv`**: Contains information about job postings from various companies.  
- **`candidates_job.csv`**: Contains information about job-seeking candidates.

These datasets include fields such as:
- **Technical and non-technical skills**
- **Values**
- **Contract type**
- **Experience**
- **Education level**, and more.

---

## Methodology

The project follows a structured approach with the following key steps:

### 1. Attribute Encoding
Attributes of candidates and job postings (e.g., contract type, experience level, education level) are encoded using predefined mappings for categorical variables.

### 2. Handling Missing Values
Missing data is filled with default values to maintain the dataset's integrity.

### 3. Skill and Value Encoding
Non-technical skills and values are encoded as binary matrices to capture their presence or absence in a structured format.

### 4. Technical Skills Vectorization
Technical skills are vectorized using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique converts textual data into numerical feature vectors, emphasizing the relative importance of terms.

### 5. Feature Combination
All encoded features are combined into a single **DataFrame** for candidates and job postings, ensuring a unified representation of all attributes.

### 6. Cosine Similarity Calculation
The similarity between candidates and job postings is computed using **cosine similarity**. This measure evaluates the relevance of matches based on the similarity of their feature vectors.

---

## Technical Skill Vectorization with TF-IDF

The **TF-IDF** technique is used to vectorize technical skills. It transforms textual descriptions into numerical feature vectors while accounting for the importance of each term relative to the dataset. This ensures that rare but significant skills are appropriately weighted.

---

## Cosine Similarity Calculation

**Cosine similarity** is employed to determine the degree of match between candidates and job postings. It measures the cosine of the angle between two vectors (candidate features and job features) and is defined as:

\[
\text{Cosine Similarity} = \frac{A \cdot B}{\|A\| \|B\|}
\]

Where:
- \( A \) and \( B \) are the feature vectors for a candidate and a job posting.
- \( \|A\| \) and \( \|B\| \) are the magnitudes (norms) of these vectors.

A higher cosine similarity score indicates a better match.

---

## Conclusion

This system provides an automated and scalable solution for matching candidates with job postings by analyzing and comparing their respective features. The combination of data encoding, TF-IDF vectorization, and cosine similarity ensures precise and meaningful matches.
