# Detecting Implicit Biases in Electronic Patient Records: A Linguistic Analysis of Stigma and Disparities in Healthcare


# Data Collection

Active learning NLP Framework:

1. Clinical experts annotate a small set of high-quality data using subjective lexicons (e.g., pleasant, combative, talkative, confused, forgetful), with a multi-token phrase formed by adding 50 characters on either side of the lexicon for annotation. Each pair of annotators is assigned a batch.
   
2. Sentences where both annotators agree, determined by the Cohen Kappa score, are selected. These sentences are then used to fine-tune BERT for text classification, distinguishing between subjective and non-subjective sentiments.
   
3. Sentences with low confidence scores are returned to the user interface for annotation by clinical experts.
   
4. The BERT text classification model undergoes additional fine-tuning with the newly labeled data until achieving satisfactory performance on the F1-score.


The start and end tokens of the phrase, along with the text and annotation label, are extracted from the data collected. The row-id is used to match the subject-id for a comprehensive assessment of subjectivity across marginalized groups, including age groups (0-20, 20-40, 40-60, 60-80, 80+), alcoholics, disability, left against medical advice, english proficiency, gender (f/m), homelessness, iv drug users, left against medical advice, mental disorders, obese, and race (white, black, asian, hispanic, other, unknown).













