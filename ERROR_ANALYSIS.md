# Error Analysis

To ensure a robust evaluation of where the `ArvyaxModel` fails, we isolated 20% of the training dataset as a validation set and ran our inference pipeline. We sorted the errors by the **model's confidence** to identify cases where the model was "confidently wrong", as these are the most informative failures in real-world messy data.

## 10 High-Confidence Failure Cases

Below are exactly 10 examples where the model prediction heavily disagreed with the ground truth label.

| ID | Text (Snippet) | True State | Predicted State | Conf. | Key Metadata (Stress / Energy) |
|---|---|---|---|---|---|
| 621 | "honestly back to normal after. then it shifted some peace, some noise in head." | calm | neutral | 0.69 | Stress: 1, Energy: 3 |
| 789 | "at first kept thinking about work. later it changed still a bit off tbh." | focused | restless | 0.60 | Stress: 5, Energy: 4 |
| 653 | "During the session kinda calm now. Then it shifted could focus for a while." | neutral | mixed | 0.52 | Stress: 5, Energy: 1 |
| 695 | "I guess helped me plan my day." | neutral | focused | 0.48 | Stress: 3, Energy: 4 |
| 817 | "started off pretty even. i kept thinking about emails." | neutral | restless | 0.44 | Stress: 3, Energy: 1 |
| 694 | "During the session helped me plan my day. Then it shifted felt heavy." | calm | overwhelmed| 0.44 | Stress: 5, Energy: 3 |
| 809 | "During teh session that helped a little. Then it shifted breathing slowed down." | calm | mixed | 0.39 | Stress: 1, Energy: 4 |
| 110 | "I came in distracted, but I left the cafe session with a sharper mind." | focused | mixed | 0.38 | Stress: 2, Energy: 3 |
| 718 | "Honestly breathing slowed down." | focused | restless | 0.38 | Stress: 2, Energy: 1 |
| 82 | "I came in distracted, but I left the rain session with a sharper mind..." | focused | mixed | 0.37 | Stress: 2, Energy: 5 |

## Insights & Diagnosis

### 1. Ambiguous and Shifting Text
A large portion of the reflection texts contain explicit temporal shifts ("During... Then it shifted", "at first... later"). Models using simple TF-IDF bags-of-words or even sequence embeddings often average these out, leading the model to predict an intermediate state like "mixed" (Cases `653`, `809`, `110`, `82`). 

### 2. Conflicting Signals (Text vs Metadata)
Sometimes the human label reflects the text, but the metadata contradicts it drastically. 
- In **Case 789**, the text mentions "thinking about work" and "a bit off", while Stress is 5 and Energy is 4. The model accurately interprets these high-arousal negative signals as `restless`. Yet, the ground truth is labelled `focused`. This suggests the user might have felt hyper-focused in a negative, ruminating way. 
- In **Case 694**, the user explicitly says they "felt heavy", and Stress is 5. The model predicts `overwhelmed`, which is highly logical, yet the user label is `calm`. 

### 3. Noisy / Subjective Labels
Emotions are highly subjective. What one person calls `neutral` when they "helped plan my day" (Case `695`), another person (and our model) might call `focused`. When text is extremely short (e.g., Case `718`: "Honestly breathing slowed down."), the model falls back to heuristics based on energy (which is 1) and predicts `restless` or `tired`, while the ground-truth is `focused`.

## How to Improve

1. **Temporal Attention Models:** Instead of TF-IDF, using a lightweight local transformer (like `SentenceTransformers`) could help the model understand the sequence of emotions (e.g., prioritizing the state at the *end* of the session over the beginning).
2. **Confident Learning / Noise Pruning:** We implemented a basic version of this in `pipeline.py` using Out-of-Fold prediction weights. Moving forward, explicitly using symmetric cross-entropy drops or the `cleanlab` library to flip or drop highly-conflicting human labels (like Case `694`) during training would stabilize the decision boundaries.
3. **Conversational Feedback Loop:** If the system is uncertain (which our flag explicitly tracks), the UI should prompt: *"You mentioned feeling heavy but your stress looks low. Do you feel more overwhelmed or calm right now?"* This acts as active learning to personalize the embeddings.
