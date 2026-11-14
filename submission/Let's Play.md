\# Let's Play

\## Football Possession Estimation – Technical Report



\*\*Competition:\*\* More Than a Hack 2027  

\*\*Team:\*\* Let's Play  

\*\*Method:\*\* Spatial and Temporal Heuristic Possession Estimation  

\*\*Version:\*\* V3.1



\---



\## 1. Problem Definition



The objective is to estimate team possession for every frame of a football match using player tracking data and tactical homography.



Each frame receives one of four labels:



\- \*\*H:\*\* Home team (FC Barcelona).

\- \*\*A:\*\* Away team (Athletic Club).

\- \*\*D:\*\* Dead ball.

\- \*\*X:\*\* Abstention when the available evidence is insufficient.



The challenge contains 199,319 frames at 25 FPS. The system aims to produce a temporally consistent possession sequence rather than relying only on isolated frame-level predictions.



\## 2. Dataset and Alignment



The provided match is FC Barcelona vs Athletic Club, from LaLiga EA Sports 2024–25.



Dataset characteristics:



\- Total frames: 199,319.

\- Video resolution: 1920 × 1080.

\- Frame rate: 25 FPS.

\- Player tracking slots: 46.

\- Ball visibility: approximately 56.9%.

\- Missing ball observations: approximately 43.1%.

\- Tactical homography: valid across all 199,319 frames.



Official tracking metadata identifies the two teams, with 24 Barcelona tracking slots and 22 Athletic Club tracking slots.



Frame alignment follows the competition convention:



`video\_frame = frame\_id - 1`



The submission therefore uses zero-based frame indices from 0 to 199,318.



The broadcast recording is not used as a synchronized source because it is not aligned with the tactical recording.



\## 3. Feature Engineering



The pipeline transforms raw tracking information into spatial and temporal features.



\### 3.1 Spatial Features



The feature-building stage extracts:



\- Ball position and visibility.

\- Player positions and visibility.

\- Home and away team spatial centers.

\- Distances between the ball and players or team centers.

\- Nearest-player information.

\- Homography-based spatial information.

\- Additional spatial and tracking-quality indicators.



Tracking coordinates are transformed into pitch-space coordinates using the provided homography information.



\### 3.2 Temporal Features



Temporal information is added using two observation windows:



\- 1-second window.

\- 3-second window.



These features help represent short-term and longer-term patterns in the possession signal.



\### 3.3 Model Input Preparation



The preparation stage combines base, temporal, spatial, and spatial-temporal features with missing-value indicators.



The resulting model input contains:



\- 199,319 rows.

\- 206 columns, including the frame identifier.

\- 133 selected feature columns before additional missing-indicator columns.



\## 4. Possession Estimation – V3.1



V3.1 is a rule-based heuristic estimator. It does not train a supervised machine-learning classifier on possession ground truth.



\### 4.1 Direct Possession Signal



The direct signal uses the nearest tracked player to the visible ball to estimate the likely team in possession.



Official tracking metadata determines the team associated with each player.



\### 4.2 Hysteresis



A hysteresis threshold of 0.25 is applied to the distance-margin signal.



This mechanism helps reduce rapid changes in the estimated possession state when evidence is close or unstable.



The internal state is reset at the beginning of each half to avoid carrying possession state across halftime.



\### 4.3 Temporal Evidence Fusion



The estimator combines direct and temporally aggregated evidence using the following weights:



| Evidence source | Weight |

|---|---:|

| Direct signal | 0.65 |

| 1-second temporal evidence | 0.25 |

| 3-second temporal evidence | 0.10 |



The minimum evidence weight is 0.20, and the minimum confidence threshold is 0.60.



The temporal signals are derived from the nearest-player possession signal. They represent aggregated observations of the same underlying signal and should not be interpreted as fully independent evidence sources.



\### 4.4 Abstention



The estimator outputs X when the available evidence is insufficient or confidence is below the selected threshold.



It does not automatically classify missing ball observations as dead-ball situations.



Consequently, V3.1 currently produces no D predictions. This is a known limitation rather than evidence that dead-ball situations do not occur.



\## 5. Results and Diagnostics



\### 5.1 V3.1 Full Dataset Distribution



| Label | Frames | Percentage |

|---|---:|---:|

| H | 65,210 | 32.72% |

| A | 50,390 | 25.28% |

| D | 0 | 0.00% |

| X | 83,719 | 42.00% |

| \*\*Total\*\* | \*\*199,319\*\* | \*\*100%\*\* |



\### 5.2 Graded-Frame Distribution



| Label | Frames | Percentage |

|---|---:|---:|

| H | 54,718 | 34.44% |

| A | 42,608 | 26.82% |

| D | 0 | 0.00% |

| X | 61,531 | 38.73% |

| \*\*Total\*\* | \*\*158,857\*\* | \*\*100%\*\* |



\### 5.3 Evidence Diagnostics



| Diagnostic | Frames |

|---|---:|

| Direct evidence available | 94,167 |

| 1-second temporal evidence available | 117,081 |

| 3-second temporal evidence available | 124,133 |

| Insufficient evidence | 82,238 |



These diagnostics describe evidence availability, not prediction accuracy.



\### 5.4 Submission Validation



The generated submission file was checked for structural correctness:



\- 199,319 data rows.

\- Correct column names and order: `frame,label`.

\- Frame indices start at 0 and end at 199,318.

\- Frames are ordered.

\- No duplicate frame indices.

\- No missing frame or label values.

\- Labels belong to the allowed set H, A, D, X.



The CSV passed these structural checks.



No full-match ground-truth accuracy or V3.1 spell F1 score is claimed in this report.



\## 6. Evaluation Considerations



The competition emphasizes possession-spell quality through `spell\_f1\_mean`, evaluated at temporal IoU thresholds of 0.3, 0.4, 0.5, 0.6, and 0.7.



Frame-level classification alone is not sufficient. The duration and boundaries of possession spells are also important.



The public ground truth covers only a limited initial portion of the match. It is intended for pipeline validation and must not be used for repeated model tuning or private-ranking optimization.



The reported label distributions and structural validation results should not be interpreted as evidence of competitive ranking performance.



\## 7. Limitations and Future Improvements



The current V3.1 approach has several limitations:



1\. Ball observations are missing in a substantial portion of the match.

2\. Temporal evidence is derived from the same nearest-player signal and is not fully independent.

3\. Dead-ball detection is not yet implemented, resulting in zero D predictions.

4\. Abstention accounts for a significant proportion of frames.

5\. Possession transitions and short possession spells require further investigation.

6\. Full-match prediction quality has not been established using private ground truth.



Potential future improvements include:



\- Better use of player movement and team-level spatial context.

\- More robust handling of missing ball observations.

\- A dedicated dead-ball detection component.

\- Improved possession transition modeling.

\- Additional temporal consistency mechanisms.

\- Further evaluation using permitted data and competition rules.



\## 8. Reproducibility



\### Requirements



Python environment with dependencies specified in:



\- `requirements.txt`

\- `requirements-starter.txt`



Install the starter dependencies using:



```bash

pip install -r requirements-starter.txt

```



\### Data Setup



Obtain and unpack the competition dataset according to `DATA.md` and `data/README.md`.



The tactical assets must be placed in the local `data/tactical/` directory using their original filenames.



The dataset is not included in the public repository because its redistribution is restricted by the competition terms.



\### Pipeline Execution



Run the following scripts from the repository root, in order:



```bash

python build\_features\_dataset.py

python build\_temporal\_features.py

python prepare\_model\_features.py

python build\_possession\_estimator\_v3.py

```



The final candidate is generated at:



`data/processed/possession\_candidate\_v3.csv`



The submission CSV is stored separately under:



`submission/Let's Play.csv`



\### Repository Structure



```text

Let'sPlay-Repository/

├── starter/

├── evaluation/

├── data/

│   └── README.md

├── submission/

│   ├── Let's Play.csv

│   └── Let's Play.md

├── build\_features\_dataset.py

├── build\_temporal\_features.py

├── prepare\_model\_features.py

├── build\_possession\_estimator\_v3.py

├── games.py

├── requirements.txt

├── requirements-starter.txt

├── README.md

├── DATA.md

├── RULES.md

└── SUBMISSION.md

```



\## 9. Conclusion



Let's Play developed a spatial and temporal heuristic pipeline for estimating football possession from tracking data.



The V3.1 approach combines nearest-player evidence, hysteresis, temporal aggregation, confidence thresholds, and abstention.



The current implementation produces a structurally valid full-match CSV and documents its methodology and limitations.



Further improvements are required to strengthen dead-ball detection, reduce uncertainty, and improve possession-spell boundaries.



\---



\*\*Team:\*\* Let's Play  

\*\*Project:\*\* More Than a Hack 2027  

\*\*Estimator:\*\* V3.1

