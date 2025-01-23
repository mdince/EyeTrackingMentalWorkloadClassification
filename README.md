Mental Workload Classification with Eye Tracking Data

Dataset: Eye-tracking data from the COLET dataset (.mat).

Citation and link: https://zenodo.org/records/7766785
Ktistakis, E., Skaramagkas, V., Manousos, D., Tachos, N. S., Tripoliti, E., Fotiadis, D. I., & Tsiknakis, M. (2022). Colet: A dataset for cognitive workload estimation based on eye-tracking. Computer Methods and Programs in Biomedicine, 106989. https://doi.org/10.1016/j.cmpb.2022.106989

Gaze data (e.g., norm_pos_x, norm_pos_y, gaze_point_3d_x)
Pupil data (e.g., diameter, model_confidence)
Blink data (blink_count, blink_duration)

Data Processing:
Handling missing values (NaN) and calculating average values.
Frequency analysis using FFT to calculate energy values in low and high-frequency ranges (low_freq_energy, high_freq_energy).

Task Categorization:
Tasks are categorized into Low, Medium, and High levels based on annotation data.

Modeling:
The extracted features are modeled using a Random Forest Classifier, achieving an accuracy of 0.79.

Additional Steps:
Data balancing through SMOTE (Synthetic Minority Oversampling Technique).
Normalization using StandardScaler.

Model evaluation with metrics including Accuracy, Precision, Recall, and F1-Score.

![Screenshot 2025-01-13 195843](https://github.com/user-attachments/assets/2b28dc2e-6bd6-40d4-9070-f007ecee3f42)
![Screenshot 2025-01-13 195120](https://github.com/user-attachments/assets/d37373cc-a1e0-482b-8e65-c473779d7979)
![Screenshot 2025-01-13 195156](https://github.com/user-attachments/assets/6cecfa02-9a26-46f7-97ad-cfbfa26559a3)
