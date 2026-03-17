# Edge / Offline Plan (On-Device Inference)

The goal of this pipeline is to operate seamlessly on mobile or low-power embedded edge devices, providing intelligent inferences from messy human reflections without relying on cloud APIs. At ArvyaX, understanding the human emotional state in real-time, completely offline, is crucial for both latency and data privacy.

## Deployment Approach: On-Device ML Pipeline

Our existing solution (`pipeline.py`) relies on `scikit-learn` and `XGBoost`. To port this entirely to a mobile device (iOS/Android):

1. **Model Conversion to ONNX or CoreML / TFLite**
   - We will convert both the `XGBClassifier` (Emotion) and `XGBRegressor` (Intensity) into the **ONNX (Open Neural Network Exchange)** format. ONNX Runtime provides extremely fast and lightweight execution locally across iOS, Android, and web.
   - Using `hummingbird-ml` or `skl2onnx`, the `ColumnTransformer` (TF-IDF vectorizer, Scaler, Imputer, and One-Hot Encoders) can also be directly baked into the ONNX graph. This means the app simply passes raw text and floats into the ONNX session, entirely bypassing a heavy Python backend.

2. **Decision Engine Logic**
   - The rule-based engine mapping predictions to actionable `what_to_do` and `when_to_do` is extremely lightweight. This can be natively rewritten in Swift (iOS) or Kotlin (Android), evaluating the outputs from the ONNX session in <1ms. Uncertainty flags (probability entropy and margin) can also be easily calculated on-device.

3. **In-Memory Caching and SQLite**
   - Context signals like `sleep_hours` and `stress_level` will be pulled directly from Apple HealthKit or Google Fit locally.
   - A local SQLite database will track the user's `previous_day_mood` allowing zero-network state tracking.

## Optimizations & Tradeoffs

### Model Size & Footprint
- **Size Optimization:** A scikit-learn TF-IDF model with 2500 max features and an XGBoost model of depth 4 and 150 trees typically take up **< 5 MB** combined. This is a dramatic reduction compared to LLMs (which require 1-3 GBs memory).
- **Latency:** Inference on an iPhone 13 running ONNX Runtime on a 5MB XGBoost model takes under **3 milliseconds**. This enables real-time guidance while typing.
- **Tradeoff - Vocabulary Freezing:** Because TF-IDF uses a static bag-of-words, newly trending slang or emojis won't be inherently understood until a model update is pushed out via App Store updates or background OTA downloads. A tradeoff here is potentially using a tiny pre-trained character-CNN or `all-MiniLM-L6-v2` quantization, which pushes the size to ~45 MB but handles typos significantly better. 

### Robustness & Messy Inputs
- **Very short texts (“ok”, “fine”):** Our model heavily incorporates metadata (`stress_level`, `energy_level`). When text is meaningless or completely absent, the Decision Engine falls back to checking physiological metrics. If text is tiny, the Uncertainty Flag triggers, and the app interface could gently prompt: *"Could you elaborate just a bit more?"*
- **Missing Data:** Our pipeline handles NaNs via median and frequent imputation dynamically. If Apple Health fails to sync sleep data, the engine naturally ignores it and weighs the text higher.
- **Contradictory Inputs:** When text feels highly "calm" but Stress is 5, the model's prediction probability drops, raising the `uncertain_flag`. The decision engine then defaults to safer, short-term calming interventions (like `box_breathing`) instead of deep cognitive tasks (`deep_work`), acting on the principle of "First, do no harm".
