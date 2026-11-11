\# Formative 2 – Multimodal Data Preprocessing and Authentication System



\*\*Facial Verification | Voice Verification | Product Recommendation\*\*



\## 1. Project Overview



This project focuses on developing a multimodal system that combines facial verification, voice verification and product recommendation.



The system is designed to verify a user's identity through their face and voice before predicting the product category they are most likely to purchase.



The project is divided into three main tasks, with each team member responsible for developing one component:



| Task | Component | Main responsibility |

|---|---|---|

| Task 1 – Member 1 | Data Merge, Preprocessing and Product Recommendation | Prepare customer transaction and social-profile data, develop features and build a product prediction model. |

| Task 2 – Member 2 | Image Preprocessing and Facial Verification | Process facial images, extract facial embeddings and verify enrolled users. |

| Task 3 – Member 3 | Audio Processing and Voice Verification | Process voice recordings, extract audio features, generate voice profiles and verify speakers. |



The intended system workflow is:



\*\*Facial Verification → Voice Verification → Product Recommendation\*\*



The product prediction component is only called after both identity verification stages have succeeded.



\### Project Objectives



The overall objectives are to:



\- Collect and prepare different types of data, including customer information, facial images and audio recordings.

\- Clean, preprocess and explore the collected datasets.

\- Apply suitable feature extraction and feature engineering techniques.

\- Develop a facial verification component for identifying enrolled users.

\- Develop a voice verification component for checking speaker identity.

\- Build a machine learning model that predicts a customer's likely product category.

\- Evaluate the individual components using appropriate metrics.

\- Prepare the components for integration into one multimodal system.



\## 2. Project Structure



The repository contains the notebooks, source code, datasets, model files, documentation and evaluation outputs for the three tasks.



The main project structure is organised as follows:



```text

Formative-2-Multimodal-System/

│

├── data/

│   ├── raw/

│   │   ├── customer\_transactions.xlsx

│   │   ├── customer\_social\_profiles.xlsx

│   │   ├── images/

│   │   │   ├── member\_1/

│   │   │   ├── member\_2/

│   │   │   └── member\_3/

│   │   │

│   │   └── audio/

│   │       ├── member\_1/

│   │       ├── member\_2/

│   │       └── member\_3/

│   │

│   ├── test\_unknown/

│   │   └── person\_N/

│   │

│   └── processed/

│       ├── merged\_customer\_data.csv

│       ├── audio/

│       ├── audio\_augmented/

│       ├── audio\_features.csv

│       ├── original\_file\_mapping.csv

│       └── unknown\_file\_mapping.csv

│

├── models/

│   ├── product\_model.joblib

│   ├── voice\_profiles.npz

│   └── face/

│       ├── config/

│       ├── templates/

│       └── metadata/

│

├── src/

│   ├── product\_model.py

│   ├── face\_model.py

│   ├── image\_preprocessing.py

│   ├── feature\_extraction.py

│   ├── evaluation.py

│   ├── unknown\_testing.py

│   └── voice\_model.py

│

├── notebooks/

│   ├── 01\_tabular\_preprocessing.ipynb

│   ├── 02\_image\_preprocessing.ipynb

│   └── 03\_audio\_preprocessing.ipynb

│

├── outputs/

│   ├── figures/

│   ├── evaluation/

│   ├── augmented\_images/

│   └── processed\_images/

│

├── docs/

│   ├── dataset\_documentation.md

│   ├── methodology.md

│   ├── evaluation\_report.md

│   └── integration\_guide.md

│

├── requirements.txt

├── .gitignore

└── README.md

```



\*Note: This is the overall structure showing how the three components fit into the repository. Some files and directories are generated when the notebooks are executed. The actual repository may contain additional task-specific files.\*



\## 3. Task 1 – Data Merge, Preprocessing and Product Recommendation



\*\*Responsible member: Member 1\*\*



\### 3.1 Overview



Task 1 focuses on developing the product-recommendation component of the multimodal system.



The work involves combining customer transaction data with customer social-profile data, exploring and cleaning the merged dataset, engineering useful features and training a machine learning model to predict the product category a customer is likely to purchase.



The prediction module is designed to receive a customer ID and return the predicted product category together with the probabilities for each category.



The prediction function is:



```python

predict\_product(customer\_id)

```



It is intended to be called only after the facial and voice verification stages have succeeded.



\### 3.2 Dataset



Two datasets were used:



| Dataset | Original records | Columns | Customers |

|---|---:|---:|---:|

| Customer transactions | 150 | 6 | 75 |

| Customer social profiles | 155 | 5 | 84 |



The datasets contain customer transaction information and social-profile information.



The customer IDs were initially represented differently in the two files. They were standardised into a common `customer\_id` field before merging.



\### 3.3 Data Cleaning and Merging



The following preprocessing steps were carried out:



\- Standardising customer IDs across both datasets.

\- Converting data types, including IDs, amounts and dates.

\- Standardising text categories.

\- Checking the valid ranges of ratings, engagement and purchase interest.

\- Removing five exact duplicate rows from the social-profile dataset.

\- Handling missing values within the machine learning pipeline rather than filling them during initial cleaning.



The two datasets contained multiple records for some customers. A direct join would have duplicated transactions and their associated product labels.



To avoid this, the social-profile records were first aggregated to one row per customer. The aggregated information was then joined to the transaction dataset using a left join.



The resulting merged dataset retained all 150 transaction records.



| Merge result | Records |

|---|---:|

| Customers with social-profile information | 117 (78%) |

| Customers without social-profile information | 33 (22%) |

| Total transaction records retained | 150 |



Merge validation checks confirmed that the transaction count was preserved, transaction IDs remained unique, join keys were valid and the target column contained no missing values.



\### 3.4 Exploratory Data Analysis



Exploratory data analysis was conducted to understand the dataset before modelling.



The analysis included:



\- Summary statistics and data types.

\- Missing-value analysis.

\- Product-category distribution.

\- Purchase amount distribution and outlier inspection.

\- Correlation analysis.

\- Social platform distribution across product categories.

\- Ratings and purchase interest by product category.



The five product categories were relatively balanced:



| Product category | Percentage |

|---|---:|

| Sports | 23.3% |

| Electronics | 22.7% |

| Clothing | 18.7% |

| Groceries | 18.7% |

| Books | 16.7% |



The analysis showed generally weak relationships between the available features and the product category.



\### 3.5 Feature Engineering



Features were developed from three main groups:



\*\*Social-profile features\*\*

\- Average engagement.

\- Average purchase interest.

\- Positive, neutral and negative sentiment proportions.

\- Most frequent social platform.

\- Number of platforms and social records.

\- Whether a customer has a social profile.

\- Engagement-interest interaction.



\*\*Time-based features\*\*

\- Purchase month.

\- Purchase weekday.

\- Weekend indicator.



\*\*Customer purchase-history features\*\*

\- Number of previous purchases.

\- Average amount from previous purchases.

\- Average previous rating.

\- Previous purchase counts by product category.



Purchase-history features were calculated using only transactions that occurred before the current transaction. This helped prevent information from future purchases from entering the model.



Features that were near-constant or highly correlated were removed using the feature-selection process.



The final model used \*\*20 features\*\*.



\### 3.6 Model Development and Evaluation



The dataset was divided into training and testing sets using `GroupShuffleSplit`, grouping records by customer. This ensured that the same customer did not appear in both sets.



| Dataset | Records | Customers |

|---|---:|---:|

| Training | 125 | 60 |

| Testing | 25 | 15 |



The following models were evaluated:



\- Majority-class baseline.

\- Logistic Regression.

\- Random Forest.

\- XGBoost.



The preprocessing steps were included inside the scikit-learn pipelines so that imputers, scalers and encoders were fitted using training data only.



Evaluation included:



\- Five-fold customer-grouped cross-validation.

\- Grouped hold-out testing.

\- Repeated stratified grouped cross-validation.

\- Top-k accuracy.

\- Permutation testing.



The models were evaluated using accuracy, macro F1-score and log loss.



\*\*Main results\*\*



| Model | CV Accuracy | CV Macro F1 | Test Accuracy | Test Macro F1 |

|---|---:|---:|---:|---:|

| Baseline | 0.173 | 0.058 | 0.160 | 0.055 |

| Logistic Regression | 0.227 | 0.189 | 0.160 | 0.116 |

| Random Forest | 0.207 | 0.149 | 0.240 | 0.185 |

| XGBoost | 0.220 | 0.183 | 0.160 | 0.125 |



Logistic Regression was selected based on cross-validated macro F1-score.



However, the model's performance was close to the baseline and random guessing across the five product categories.



The permutation test also showed that the model's performance could not be distinguished from chance using the available data:



\- Accuracy p-value: 0.29.

\- Macro F1 p-value: 0.32.



The results indicate that the available customer information does not provide enough predictive signal for reliable product-category prediction.



The model is therefore presented as an academic demonstration rather than a validated recommendation system.



\### 3.7 Prediction Module



The prediction module is located at:



```text

src/product\_model.py

```



It provides the `predict\_product(customer\_id)` interface.



Example:



```python

from product\_model import predict\_product



result = predict\_product(100)

print(result)

```



The returned result contains:



\- Customer ID.

\- Predicted product category.

\- Prediction confidence.

\- Probability assigned to each product category.



The module also provides a function for listing valid customer IDs.



Command-line usage from the repository root:



```bash

\# List valid customer IDs

python src/product\_model.py --list



\# Predict for customer 100

python src/product\_model.py --customer 100



\# Retrain the model

python src/product\_model.py --retrain

```



The saved model is refitted on all 150 transaction records after evaluation. The cross-validation and hold-out results remain the more appropriate estimates of model performance.



\### 3.8 Task 1 Files



| File | Purpose |

|---|---|

| `notebooks/01\_tabular\_preprocessing.ipynb` | Data cleaning, EDA, feature engineering and modelling |

| `src/product\_model.py` | Product prediction module |

| `data/raw/customer\_transactions.xlsx` | Original transaction data |

| `data/raw/customer\_social\_profiles.xlsx` | Original social-profile data |

| `data/processed/merged\_customer\_data.csv` | Merged and feature-engineered dataset |

| `models/product\_model.joblib` | Saved product prediction model |

| `outputs/evaluation/` | Model comparison and evaluation results |

| `outputs/figures/` | EDA and evaluation visualisations |



Further details are available in the Task 1 notebook and its evaluation outputs.



\## 4. Task 2 – Image Preprocessing and Facial Verification



\*\*Responsible member: Member 2\*\*



\### 4.1 Overview



Task 2 develops the facial image-processing and verification component of the multimodal system.



The work covers collecting and organising facial images, preprocessing, face detection, alignment, augmentation, feature extraction, template creation and identity verification.



A pretrained \*\*FaceNet512 model\*\*, accessed through DeepFace, is used to extract facial embeddings. These embeddings are compared with the stored templates of enrolled members using cosine similarity.



The main verification function is:



```python

verify\_face(image\_path)

```



It determines whether an input face belongs to an enrolled member based on a fixed similarity threshold.



\### 4.2 Dataset



The enrolled dataset contains nine original images from three team members. Each member provided three facial expressions:



\- Neutral.

\- Smiling.

\- Surprised.



Four augmentation techniques were applied to each original image:



1\. Rotation (+8°).

2\. Horizontal flipping.

3\. Grayscale conversion.

4\. Brightness adjustment (×1.2).



This generated 36 augmented images, giving a total of 45 enrolled image records.



A separate dataset containing nine images from three people who were not enrolled was prepared for unknown-person testing.



| Dataset | Number of images | Purpose |

|---|---:|---|

| Original enrolled images | 9 | Reference images for enrolled members |

| Augmented images | 36 | Additional variations of enrolled images |

| Total enrolled records | 45 | Preprocessing and feature extraction |

| Unknown-person images | 9 | Testing rejection of people who are not enrolled |



The unknown-person images were kept separate from the enrolled dataset and were not used to create or update facial templates.



\### 4.3 Image Preprocessing



The image-processing workflow included:



1\. Loading and inspecting the original images.

2\. Correcting image orientation using EXIF information.

3\. Converting images to RGB.

4\. Resizing large images to a maximum dimension of 1024 pixels.

5\. Detecting faces using RetinaFace.

6\. Identifying facial landmarks and aligning detected faces.

7\. Cropping and resizing aligned faces to 160 × 160 pixels.

8\. Applying image augmentation.

9\. Extracting facial embeddings using FaceNet512.



The FaceNet512 model produces a \*\*512-dimensional embedding\*\* for each successfully processed image.



The extracted features are stored in:



```text

image\_features.csv

```



\### 4.4 Facial Verification



The verification process compares an input image's facial embedding with the saved templates of enrolled members.



Cosine similarity is used to measure the similarity between the input embedding and each stored template.



The system uses a fixed threshold of \*\*0.70\*\*:



\- Similarity score greater than or equal to 0.70: accept.

\- Similarity score below 0.70: reject.



The main implementation is located in:



```text

src/face\_model.py

```



Example usage:



```python

from src.face\_model import verify\_face



result = verify\_face("some\_photo.jpg")

print(result)

```



The result includes the verification decision, user ID, similarity score, message and additional information about the verification process.



Unknown faces return `verified=False` and `user\_id=None`.



\### 4.5 Model Evaluation



The evaluation covered three areas:



\*\*Initial evaluation\*\*



The initial evaluation examined genuine attempts from enrolled members and leave-one-member-out probes. It checked whether enrolled members were correctly recognised and whether excluded images were rejected.



\*\*Unknown-person testing\*\*



A separate test was conducted using nine images from three people who had never been enrolled.



The images passed through the same preprocessing and feature-extraction pipeline before being compared with the saved member templates.



The results were:



| Metric | Result |

|---|---:|

| Unknown people tested | 3 |

| Unknown images tested | 9 |

| Successfully processed | 9 |

| Correctly rejected | 9 |

| False acceptances | 0 |

| Highest similarity score | 0.6265 |

| Acceptance threshold | 0.70 |



All nine unknown-person images were rejected. The highest similarity score of 0.6265 was below the acceptance threshold.



The test was carried out without adding unknown-person images to the enrolled feature table or updating the saved templates.



\*\*Combined evaluation\*\*



The combined evaluation considered the initial samples and the additional unknown-person samples.



The evaluation included accuracy, precision, recall, F1-score, False Acceptance Rate (FAR), False Rejection Rate (FRR), confusion matrices and confidence intervals.



The metrics were interpreted according to the type of sample being evaluated. The unknown-person test contains no genuine attempts, so FRR, precision, recall and F1-score are not applicable to that test on its own.



\*\*Limitations\*\*



The unknown-person test contains nine images from only three individuals. Images from the same person are not fully independent tests. The results are therefore encouraging for the controlled dataset but do not establish real-world reliability or security.



The project also does not include liveness or spoof detection.



No training or validation loss is reported because the pretrained FaceNet512 model was used for feature extraction and was not trained or fine-tuned as part of this task.



\### 4.6 Task 2 Files



| File | Purpose |

|---|---|

| `notebooks/02\_image\_preprocessing.ipynb` | Image preprocessing and facial verification workflow |

| `src/face\_model.py` | Face verification implementation |

| `src/image\_preprocessing.py` | Image preprocessing |

| `src/feature\_extraction.py` | Facial feature extraction |

| `src/evaluation.py` | Evaluation functions |

| `src/unknown\_testing.py` | Unknown-person testing |

| `data/raw/images/` | Original enrolled images |

| `data/test\_unknown/` | Images of people who are not enrolled |

| `models/face/` | Face model configuration, templates and metadata |

| `outputs/augmented\_images/` | Augmented facial images |

| `outputs/processed\_images/` | Aligned and resized facial images |

| `outputs/evaluation/` | Evaluation results and reports |



Additional documentation is available in:



\- `docs/dataset\_documentation.md`

\- `docs/methodology.md`

\- `docs/evaluation\_report.md`

\- `docs/integration\_guide.md`



\## 5. Task 3 – Audio Processing and Voice Verification



\*\*Responsible member: Member 3\*\*



\### 5.1 Overview



Task 3 develops the audio-processing and voice-verification component of the multimodal system.



The work covers audio collection, preprocessing, augmentation, feature extraction, voice profile generation and speaker verification.



The objective is to enable the system to check a user's identity through their voice before allowing the process to continue to product prediction.



\### 5.2 Audio Dataset



Six original audio recordings were collected from the three team members.



Each member provided two recordings containing the following phrases:



\- "Yes, approve"

\- "Confirm transaction"



The original recordings were organised by team member.



```text

data/

└── raw/

&#x20;   └── audio/

&#x20;       ├── member\_1/

&#x20;       ├── member\_2/

&#x20;       └── member\_3/

```



The original recordings had different sampling rates and audio formats. These differences were standardised during preprocessing.



\### 5.3 Audio Preprocessing



The recordings were processed to ensure consistency before feature extraction.



The preprocessing steps included:



\- Converting audio to mono.

\- Resampling audio to 16,000 Hz.

\- Removing silent portions through silence trimming.

\- Normalising audio amplitude.

\- Saving processed recordings in WAV format using 16-bit PCM encoding.



The processed recordings are stored in:



```text

data/processed/audio/

```



\### 5.4 Audio Augmentation



Two augmentation techniques were applied to each original recording to introduce variation into the dataset.



\*\*Pitch shifting\*\*



The pitch of each recording was increased by one semitone to create a slightly different version of the original voice recording.



\*\*Gaussian noise addition\*\*



Gaussian background noise was added using a target Signal-to-Noise Ratio (SNR) of 25 dB. This simulates background noise that may occur during voice verification.



The augmented recordings are stored in:



```text

data/processed/audio\_augmented/

```



A total of 12 augmented recordings were generated from the six original recordings.



\### 5.5 Audio Feature Extraction



Audio features were extracted to represent important characteristics of each recording numerically.



The following features were used:



| Feature | Description |

|---|---|

| MFCC | Represents important characteristics of the voice's spectral envelope. |

| Spectral Rolloff | Measures the frequency below which a specified proportion of spectral energy is concentrated. |

| RMS Energy | Represents the overall signal energy. |

| Spectral Centroid | Represents the centre of mass of the audio spectrum. |

| Zero Crossing Rate | Measures how frequently the audio signal changes sign. |



For each feature, the mean and standard deviation were calculated.



For MFCC, 13 coefficients were used, with the mean and standard deviation calculated for each coefficient. This produced 26 MFCC features.



The remaining four audio features each contributed two statistical values, giving a total of \*\*34 numerical features per recording\*\*.



The resulting dataset contains:



\- 18 audio records (6 original and 12 augmented).

\- 34 numerical audio features per record.

\- Metadata identifying the member, filename, source recording and augmentation type.



The extracted features are saved in:



```text

data/processed/audio\_features.csv

```



\### 5.6 Voice Profile Generation



Voice profiles were created for each enrolled team member using their original audio recordings.



The profiles are based on the 26 MFCC statistical features extracted from the original recordings.



Each member's profile represents the average voice characteristics of their available original recordings.



Only original recordings were used to create the voice profiles. Augmented recordings were not treated as independent identity samples.



The saved voice profiles are stored in:



```text

models/voice\_profiles.npz

```



The archive contains:



\- Enrolled member IDs.

\- Normalised voice profile vectors.

\- The selected verification threshold.



\### 5.7 Voice Verification Model



The voice verification implementation is located in:



```text

src/voice\_model.py

```



The `VoiceVerifier` class provides the functionality required to verify a speaker against an enrolled identity.



Its main operations include:



1\. Loading the enrolled voice profiles.

2\. Extracting MFCC features from a new audio recording.

3\. Comparing the extracted features with the claimed member's voice profile.

4\. Calculating cosine similarity.

5\. Comparing the similarity score against the verification threshold.

6\. Returning the verification result.



The model returns the verification decision, claimed member ID, similarity score, threshold and a corresponding message.



\*\*Verification threshold\*\*



A provisional threshold of approximately \*\*0.97165\*\* was selected based on the genuine and impostor similarity scores from the available recordings.



This threshold is experimental and requires further validation with a larger and more diverse voice dataset.



\### 5.8 Model Testing and Evaluation



The voice verification module was tested using genuine, incorrect-identity and unknown-identity scenarios.



| Test scenario | Expected result | Actual result |

|---|---|---|

| Genuine voice claiming the correct identity | Accept | Passed |

| Genuine voice claiming another member's identity | Reject | Passed |

| Voice claiming an unknown member ID | Reject | Passed |



The leave-one-recording-out evaluation correctly classified all six available trials.



These results show that the implementation produced the expected outcomes with the available recordings. However, the dataset is small and does not establish how the model would perform with more speakers, different phrases or real-world background noise.



\### 5.9 Task 3 Files



| File | Purpose |

|---|---|

| `notebooks/03\_audio\_preprocessing.ipynb` | Audio preprocessing and voice verification workflow |

| `src/voice\_model.py` | Voice verification implementation |

| `data/raw/audio/` | Original audio recordings |

| `data/processed/audio/` | Preprocessed audio recordings |

| `data/processed/audio\_augmented/` | Augmented audio recordings |

| `data/processed/audio\_features.csv` | Extracted audio features |

| `models/voice\_profiles.npz` | Saved enrolled voice profiles |



\### 5.10 Technologies and Libraries



The audio component uses:



\- \*\*Librosa:\*\* Audio loading, preprocessing, augmentation and feature extraction.

\- \*\*NumPy:\*\* Numerical operations and voice profile calculations.

\- \*\*Pandas:\*\* Organising and saving extracted audio features.

\- \*\*SoundFile:\*\* Reading and writing WAV audio files.

\- \*\*Scikit-learn:\*\* Similarity calculations and evaluation utilities.

\- \*\*Matplotlib:\*\* Audio waveform and spectrogram visualisations.



\## 6. Technologies Used Across the Project



The project combines data analysis, image processing, audio processing and machine learning technologies.



| Technology / Library | Main use |

|---|---|

| Python | Main programming language |

| Pandas | Data manipulation and analysis |

| NumPy | Numerical operations |

| Scikit-learn | Machine learning pipelines, feature processing and evaluation |

| Matplotlib | Data and model visualisations |

| Seaborn | Statistical visualisations |

| Joblib | Saving and loading the product prediction model |

| OpenPyXL | Reading Excel datasets |

| XGBoost | Product-category model comparison |

| DeepFace | Facial feature extraction and model integration |

| FaceNet512 | Facial embedding extraction |

| RetinaFace | Face detection and facial landmark detection |

| TensorFlow / tf-keras | Deep learning framework |

| OpenCV | Image processing |

| Pillow | Image loading and manipulation |

| Librosa | Audio processing and feature extraction |

| SoundFile | Audio file reading and writing |

| Jupyter Notebook | Development and experimentation |

| Google Colab | Cloud-based notebook execution |



The project dependencies are listed in `requirements.txt`.



\## 7. Installation and Setup



\### 7.1 Requirements



The project requires:



\- Python 3.10 or newer is recommended.

\- The packages listed in `requirements.txt`.

\- Jupyter Notebook or a compatible notebook environment.

\- An internet connection for downloading required pretrained model weights during the initial setup of the facial-verification component.



The exact package versions should be taken from the repository's `requirements.txt`.



\### 7.2 Local Installation



Clone the repository:



```bash

git clone <repository-url>

```



Navigate to the project folder:



```bash

cd Formative-2-Multimodal-System

```



Create a virtual environment:



```bash

python -m venv .venv

```



Activate the environment.



\*\*Windows – Git Bash\*\*



```bash

source .venv/Scripts/activate

```



\*\*Windows – PowerShell\*\*



```powershell

.venv\\Scripts\\Activate.ps1

```



\*\*Windows – Command Prompt\*\*



```cmd

.venv\\Scripts\\activate

```



\*\*macOS / Linux\*\*



```bash

source .venv/bin/activate

```



Upgrade pip and install the dependencies:



```bash

python -m pip install --upgrade pip

pip install -r requirements.txt

```



If Jupyter is not installed:



```bash

pip install jupyter

```



\### 7.3 Running in Google Colab



The individual notebooks can also be opened and executed in Google Colab.



1\. Upload the project files or open the relevant notebook from the repository.

2\. Make sure the required datasets are available in the expected directories.

3\. Run the notebook cells in order.

4\. Allow the notebook to install or load the required dependencies.

5\. Check the generated outputs and evaluation results after execution.



Task 1 supports Google Drive for accessing its datasets and saving outputs. Task 2 also supports an optional Google Drive setup through its `USE\_DRIVE` setting.



\## 8. Running the Individual Tasks



Each task has its own notebook and can be executed independently.



| Task | Notebook | Main output |

|---|---|---|

| Task 1 | `notebooks/01\_tabular\_preprocessing.ipynb` | Merged customer data, product model and evaluation results |

| Task 2 | `notebooks/02\_image\_preprocessing.ipynb` | Processed facial images, embeddings, templates and verification results |

| Task 3 | `notebooks/03\_audio\_preprocessing.ipynb` | Processed audio, extracted features and voice profiles |



For each task:



1\. Open the corresponding notebook.

2\. Select the Python environment containing the required dependencies.

3\. Run the notebook cells in order.

4\. Review the generated outputs and evaluation results.



Running the notebooks again may regenerate or overwrite saved models, datasets and evaluation outputs. Make sure the latest outputs are reviewed before committing changes to GitHub.



\### 8.1 Task 1 – Product Prediction



The product prediction module can be tested from the repository root:



```bash

\# List valid customer IDs

python src/product\_model.py --list



\# Predict for a customer

python src/product\_model.py --customer 100



\# Retrain the model

python src/product\_model.py --retrain

```



\### 8.2 Task 2 – Facial Verification



The facial-verification function can be called using:



```python

from src.face\_model import verify\_face



result = verify\_face("some\_photo.jpg")

print(result)

```



The image should contain a detectable face. The result indicates whether the input was successfully verified against an enrolled member.



\### 8.3 Task 3 – Voice Verification



The voice-verification functionality is provided through the `VoiceVerifier` class in:



```text

src/voice\_model.py

```



The notebook `03\_audio\_preprocessing.ipynb` documents the audio-processing workflow, voice profile generation and evaluation. The saved profiles are loaded by the verification implementation.



\## 9. Overall Evaluation and Limitations



Each component was evaluated using methods appropriate to its task.



| Component | Evaluation approach | Main observation |

|---|---|---|

| Product recommendation | Customer-grouped cross-validation, hold-out testing, repeated CV, top-k accuracy and permutation testing | Performance was close to chance because the available features provided weak predictive information. |

| Facial verification | Genuine, leave-one-member-out and unknown-person testing | All nine unknown-person images in the recorded test were rejected using the 0.70 threshold. |

| Voice verification | Genuine, incorrect-identity and unknown-identity tests; leave-one-recording-out evaluation | All six available evaluation trials were correctly classified. |



\### 9.1 Dataset Limitations



The datasets used in this project are relatively small:



\- Task 1 uses 150 customer transactions from 75 customers.

\- Task 2 uses nine original enrolled images and nine unknown-person images from three people.

\- Task 3 uses six original audio recordings from three team members.



These dataset sizes limit how confidently the results can be generalised to new users and real-world conditions.



\### 9.2 Model Limitations



The product prediction model performs close to chance, and its predicted probabilities are not sufficiently reliable for strong recommendation claims.



The facial-verification evaluation is limited by the small number of enrolled and unknown individuals. It also does not include liveness or spoof detection.



The voice-verification threshold is provisional and was selected using a small number of genuine and impostor recordings. Further testing is required to establish how well it performs with different speakers, recording devices, phrases and background conditions.



\### 9.3 Future Improvements



Possible improvements include:



\- Collecting more customer transactions and more informative customer features.

\- Improving the product prediction model through further feature development and validation.

\- Testing facial verification with more participants, different lighting conditions, camera angles and independent recording sessions.

\- Adding liveness and spoof detection to the facial-verification component.

\- Collecting more voice recordings from a wider range of speakers.

\- Testing voice verification with different phrases, recording devices and background noise.

\- Further validating the similarity thresholds used by the face and voice models.

\- Completing and testing the integration of all three components.



\## 10. Multimodal System Integration



The three tasks provide the main components of the proposed multimodal system.



Their intended roles are:



\*\*Stage 1 – Facial Verification\*\*



The system receives a facial image and checks it against the stored facial templates. The user proceeds only if the face is successfully verified.



\*\*Stage 2 – Voice Verification\*\*



The system receives a voice recording and compares its extracted features with the voice profile associated with the claimed member. The user proceeds only if the voice is successfully verified.



\*\*Stage 3 – Product Recommendation\*\*



After both verification stages succeed, the system uses the authenticated customer's ID to call:



```python

predict\_product(customer\_id)

```



The product model then returns the predicted product category and the associated class probabilities.



The product prediction component accepts customer IDs from the transaction and social-profile data. The team must agree on the mapping between authenticated members and customer IDs before demonstrating the complete workflow.



\*\*Integration status:\*\* The three task-specific components and their individual workflows are documented in this repository. The end-to-end integration should be considered complete only after the face, voice and product-prediction modules have been connected and tested together.



\## 11. Privacy and Data Handling



This project uses different types of personal data, including customer transaction records, social-profile information, facial images, audio recordings and biometric features.



These data must be handled responsibly.



The following precautions should be observed:



\- Use the datasets only for the intended academic project.

\- Obtain appropriate consent before collecting or using facial images and voice recordings.

\- Restrict access to identifiable personal and biometric information.

\- Avoid sharing raw facial images, audio recordings, embeddings or voice templates unnecessarily.

\- Store sensitive files securely.

\- Do not upload biometric data to external services without appropriate consent and approval.

\- Follow applicable data protection requirements.

\- Remove personal and biometric data when it is no longer needed.



The repository's `.gitignore` should exclude sensitive datasets, biometric templates, feature files and other generated files that should not be publicly shared. Review the files being committed before pushing changes to GitHub.



\## 12. Conclusion



Formative 2 brings together three areas of data processing and machine learning: customer data analysis and product recommendation, facial image processing and verification, and audio processing and voice verification.



Task 1 prepares customer transaction and social-profile data and develops a model for predicting product categories. Task 2 processes facial images, extracts embeddings and checks whether a person is enrolled. Task 3 processes voice recordings, extracts audio features and checks whether a speaker matches an enrolled voice profile.



Together, these components provide the foundation for a multimodal system in which facial and voice verification are performed before product prediction.



The individual evaluations show that the components can perform their intended operations on the available datasets. However, the small dataset sizes and the limitations observed during evaluation mean that the results should be treated as academic experimental findings rather than proof of real-world performance.



Further data collection, model improvement and end-to-end integration testing are needed to strengthen the system.



\---



\*\*Formative 2 – Multimodal Data Preprocessing and Authentication System\*\*



\*\*Project components:\*\* Data Merge and Product Recommendation | Image Preprocessing and Facial Verification | Audio Processing and Voice Verification



\*\*Intended workflow:\*\* Face Verification → Voice Verification → Product Recommendation


