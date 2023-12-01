# Detecting Implicit Biases in Electronic Patient Records: A Linguistic Analysis of Stigma and Disparities in Healthcare


# Overview


In the Active Learning NLP Framework, the process begins with clinical experts annotating a small set of high-quality data using subjective lexicons (e.g., pleasant, combative, talkative, confused, forgetful). This annotation involves creating a multi-token phrase by adding 50 characters on either side of the lexicon for annotation, with each pair of annotators assigned to a specific batch. Subsequently, sentences where both annotators agree, as determined by the Cohen Kappa score, are selected. These agreed-upon sentences play a crucial role in fine-tuning BERT for text classification, enabling the model to discern between subjective and non-subjective sentiments. For sentences displaying low confidence scores, a feedback loop is initiated, and they undergo further annotation by clinical experts through the user interface. This iterative process enhances the model's adaptability to nuanced instances.

Concurrently, data extraction involves obtaining start and end tokens, along with the text and annotation label, from the collected data. The row-id is then leveraged to match the subject-id, facilitating a comprehensive assessment of subjectivity across diverse marginalized groups. These groups include age groups (0-20, 20-40, 40-60, 60-80, 80+), alcoholics, disability, those who left against medical advice, individuals with varying levels of English proficiency, and diverse gender categories (F/M). The assessment also spans homelessness, IV drug users, individuals who left against medical advice, those with mental disorders, individuals who are obese, and individuals from different racial backgrounds (White, Black, Asian, Hispanic, Other, unknown). This comprehensive approach ensures that a diverse and inclusive list of subgroups is considered in the evaluation of subjectivity.

Furthermore, the BERT text classification model undergoes additional rounds of fine-tuning using the newly labeled data until achieving satisfactory performance on the F1-score. This thorough framework ensures continuous improvement and optimization of the NLP model through active learning principles.

# Demo

<img width="881" alt="UI" src="https://github.com/danamouk/nlp-bias/assets/49573192/3d4ee34c-dce1-48a0-bfb8-74d09e0a1059">

# Example Phrase

| id                | row-id             | begin-token | end-token | lexicon | phrase                                              | label      | confidence |
|-------------------|--------------------|--------------|-----------|---------|-----------------------------------------------------|------------|------------|
| 455461-739-746    | 455461             | 739          | 746       | anxious | arouses to voice/stimulation. Becomes easily anxious. Attempting to mouth words and gesturing but d | SUBJECTIVE | 0.98       |

# Marginalized Subgroup Assessment

<img width="550" alt="bubble-plot" src="https://github.com/danamouk/nlp-bias/assets/49573192/5ce991db-6978-4ef5-a76d-d91f8cb8169a">

<img width="550" alt="intersections" src="https://github.com/danamouk/nlp-bias/assets/49573192/b903ebd2-edf8-4470-9b2c-7911532704a4">








