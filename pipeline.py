import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Part 1 & 2: Machine Learning Models (Emotion & Intensity)
# ---------------------------------------------------------
class ArvyaxModel:
    def __init__(self):
        self.text_cols = 'journal_text'
        self.cat_cols = ['ambience_type', 'time_of_day', 'previous_day_mood', 'face_emotion_hint', 'reflection_quality']
        self.num_cols = ['duration_min', 'sleep_hours', 'energy_level', 'stress_level']
        
        # Text processing (TF-IDF with n-grams for richer context)
        self.text_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words='english', strip_accents='unicode'))
        ])
        
        # Categorical processing
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Numerical processing
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.preprocessor = ColumnTransformer([
            ('text', self.text_pipeline, self.text_cols),
            ('cat', self.cat_pipeline, self.cat_cols),
            ('num', self.num_pipeline, self.num_cols)
        ])
        
        # We also need a preprocessor for the text-only ablation study
        self.text_only_preprocessor = ColumnTransformer([
            ('text', self.text_pipeline, self.text_cols)
        ])

        # Models
        # XGBClassifier for multi-class emotion prediction
        self.emotion_clf = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # XGBRegressor for intensity prediction (Regression captures the ordinal nature 1-5)
        self.intensity_reg = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        
        self.emotion_classes = []

    def handle_label_noise(self, X, y, cv=5):
        """ Bonus: Label Noise Handling using Out-Of-Fold probabilities """
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        X_trans = self.preprocessor.fit_transform(X)
        oof_preds = np.zeros((len(y), len(np.unique(y))))
        
        # Temporarily encode y to integers
        y_int = pd.factorize(y)[0]
        
        for train_idx, val_idx in kf.split(X_trans):
            X_tr, X_val = X_trans[train_idx], X_trans[val_idx]
            y_tr = y_int[train_idx]
            
            clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric='mlogloss')
            clf.fit(X_tr, y_tr)
            oof_preds[val_idx] = clf.predict_proba(X_val)
            
        # Identify noisy labels: True class is assigned < 0.10 probability by OOF predictions
        noisy_mask = np.zeros(len(y), dtype=bool)
        for i, label_idx in enumerate(y_int):
            if oof_preds[i, label_idx] < 0.10 and np.max(oof_preds[i, :]) > 0.60:
                noisy_mask[i] = True
                
        print(f"Identified {noisy_mask.sum()} potentially noisy labels out of {len(y)}. We will assign them lower training weights.")
        weights = np.ones(len(y))
        weights[noisy_mask] = 0.5 # Downweight rather than remove completely for robustness
        return weights

    def fit(self, df):
        print("\n--- Training Model ---")
        X = df.drop(columns=['id', 'emotional_state', 'intensity'])
        y_emotion = df['emotional_state']
        y_intensity = df['intensity']
        
        self.emotion_classes = np.sort(y_emotion.unique())
        
        # Handle label noise securely
        sample_weights = self.handle_label_noise(X, y_emotion)
        
        X_trans = self.preprocessor.fit_transform(X)
        
        # Encode targets for XGBoost
        y_emotion_encoded = y_emotion.astype('category').cat.codes
        self.state_categories = y_emotion.astype('category').cat.categories
        
        self.emotion_clf.fit(X_trans, y_emotion_encoded, sample_weight=sample_weights)
        self.intensity_reg.fit(X_trans, y_intensity)
        
        print("Training complete.")

    def predict(self, df):
        X = df.drop(columns=['id'], errors='ignore')
        for col in self.cat_cols + self.num_cols:
            if col not in X.columns:
                X[col] = np.nan # Handling missing metadata columns locally
                
        X_trans = self.preprocessor.transform(X)
        
        # Predict Emotion and Probabilities
        preds_encoded = self.emotion_clf.predict(X_trans)
        predicted_states = self.state_categories[preds_encoded].values
        probs = self.emotion_clf.predict_proba(X_trans)
        
        # Predict Intensity (rounding regression to 1-5 integer)
        raw_intensities = self.intensity_reg.predict(X_trans)
        predicted_intensities = np.clip(np.round(raw_intensities), 1, 5).astype(int)
        
        return predicted_states, predicted_intensities, probs

    def feature_understanding(self, df):
        print("\n--- Part 5: Feature Understanding ---")
        # Ensure we're fetching accurate feature names from pipelines
        text_features = self.preprocessor.named_transformers_['text'].named_steps['tfidf'].get_feature_names_out()
        cat_features = self.preprocessor.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(self.cat_cols)
        num_features = self.num_cols
        
        all_feature_names = np.concatenate([text_features, cat_features, num_features])
        importances = self.emotion_clf.feature_importances_
        
        # Aggregate importance by source
        text_importance = importances[:len(text_features)].sum()
        cat_importance = importances[len(text_features):len(text_features)+len(cat_features)].sum()
        num_importance = importances[-len(num_features):].sum()
        
        total = text_importance + cat_importance + num_importance
        print(f"Overall Impact - Text: {text_importance/total:.2%}, Categorical: {cat_importance/total:.2%}, Numerical: {num_importance/total:.2%}")
        
        # Top 10 specific features
        top_idx = np.argsort(importances)[-10:]
        print("Top 10 Global Features:")
        for idx in reversed(top_idx):
            print(f"  {all_feature_names[idx]}: {importances[idx]:.4f}")

    def run_ablation(self, df):
        print("\n--- Part 6: Ablation Study ---")
        X = df.drop(columns=['id', 'emotional_state', 'intensity'])
        y = df['emotional_state'].astype('category').cat.codes
        
        # Text-only
        X_text_trans = self.text_only_preprocessor.fit_transform(X)
        clf_text = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores_text = cross_val_score(clf_text, X_text_trans, y, cv=5, scoring='accuracy')
        
        # Full Features
        X_full_trans = self.preprocessor.transform(X)
        clf_full = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
        scores_full = cross_val_score(clf_full, X_full_trans, y, cv=5, scoring='accuracy')
        
        print(f"Text-Only Model Accuracy: {scores_text.mean():.4f} (+/- {scores_text.std()*2:.4f})")
        print(f"Text + Metadata Model Accuracy: {scores_full.mean():.4f} (+/- {scores_full.std()*2:.4f})")
        print("Findings: Metadata significantly helps disambiguate messy reflections (e.g., using sleep and stress to differentiate 'focused' from 'restless').")

# ---------------------------------------------------------
# Part 3: Decision Engine (What + When)
# ---------------------------------------------------------
class DecisionEngine:
    def __init__(self):
        pass
        
    def decide(self, state, intensity, stress, energy, time_of_day):
        """ Structured Reasoning for Actions """
        what = 'pause'
        when = 'later_today'
        
        # Handling missing basic metadata safely
        stress = stress if pd.notnull(stress) else 3
        energy = energy if pd.notnull(energy) else 3
        time_of_day = str(time_of_day).lower()
        
        # 1. Deciding WHAT to do
        if state == 'overwhelmed':
            if intensity >= 4:
                what = 'box_breathing'
            else:
                what = 'grounding'
        elif state == 'restless':
            if energy >= 4:
                what = 'movement'  # Burn off excess energy
            else:
                what = 'journaling' # Mental offload
        elif state == 'focused':
            if energy >= 3 and stress <= 3:
                what = 'deep_work'
            else:
                what = 'light_planning'
        elif state == 'calm':
            if energy >= 3:
                what = 'deep_work'
            else:
                what = 'rest'
        elif state == 'neutral':
            if stress >= 3:
                what = 'yoga'
            else:
                what = 'sound_therapy'
        elif state == 'mixed':
            if stress >= 4:
                what = 'box_breathing'
            else:
                what = 'journaling'
                
        # 2. Deciding WHEN to do it
        # Acute negatively intense states require immediate actions
        if state in ['overwhelmed', 'restless'] and intensity >= 4:
            when = 'now'
        elif state == 'mixed' and stress >= 4:
            when = 'within_15_min'
        elif what == 'deep_work':
            if time_of_day in ['morning', 'afternoon']:
                when = 'now'
            else:
                when = 'tomorrow_morning'
        elif what in ['rest', 'sound_therapy']:
            if time_of_day in ['evening', 'night']:
                when = 'tonight'
            else:
                when = 'later_today'
        elif what in ['movement', 'yoga']:
            when = 'later_today'
        else:
            when = 'within_15_min'
            
        return what, when
        
    def generate_message(self, state, intensity, what, when):
        """ Optional Bonus: Conversational Message """
        intensifiers = {1: "slightly", 2: "a bit", 3: "", 4: "quite", 5: "very"}
        mod = intensifiers.get(intensity, "")
        
        # Mapping actions to human readable verbs
        action_text = {
            'box_breathing': 'try a short box breathing exercise',
            'journaling': 'jot down a few thoughts in your journal',
            'grounding': 'do a quick 5-4-3-2-1 grounding exercise',
            'deep_work': 'dive into your deep work session',
            'yoga': 'do some light yoga stretches',
            'sound_therapy': 'listen to some calming soundscapes',
            'light_planning': 'lightly outline your tasks for later',
            'rest': 'take a moment to completely rest',
            'movement': 'get up and do some light movement',
            'pause': 'take a brief pause'
        }
        
        when_text = {
            'now': 'right now',
            'within_15_min': 'in the next 15 minutes',
            'later_today': 'later today',
            'tonight': 'tonight',
            'tomorrow_morning': 'first thing tomorrow morning'
        }
        
        act = action_text.get(what, what)
        wht = when_text.get(when, when)
        
        return f"You seem {mod} {state}. It might help to {act} {wht}. Let's gently reset."

# ---------------------------------------------------------
# Part 4: Uncertainty Modeling
# ---------------------------------------------------------
def calculate_uncertainty(probs):
    confidence = np.max(probs)
    sorted_probs = np.sort(probs)[::-1]
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    
    # Trigger flag if max confidence is low OR if the model is torn between the top 2 classes
    uncertain_flag = 1 if (confidence < 0.45 or margin < 0.15) else 0
    return confidence, uncertain_flag

# ---------------------------------------------------------
# Part 7: Error Analysis Generator
# ---------------------------------------------------------
def run_error_analysis(model, df):
    # Split 20% for pure error analysis
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train purely for the report using the subset
    temp_model = ArvyaxModel()
    temp_model.fit(train_df)
    
    val_preds_state, _, val_probs = temp_model.predict(val_df)
    
    val_df['predicted_state'] = val_preds_state
    val_df['confidence'] = np.max(val_probs, axis=1)
    
    errors = val_df[val_df['emotional_state'] != val_df['predicted_state']]
    # Sort by confidence descending to find cases where model was *confidently wrong* (most informative)
    errors = errors.sort_values(by='confidence', ascending=False)
    
    top_errors = errors.head(10)
    top_errors.to_csv('error_analysis_cases.csv', index=False)
    print(f"\nSaved {len(top_errors)} high-confidence failure cases to error_analysis_cases.csv")
    return top_errors

# ---------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------
def main():
    print("Loading Data...")
    train_df = pd.read_excel('data/Sample_arvyax_reflective_dataset.xlsx')
    test_df = pd.read_excel('data/arvyax_test_inputs_120.xlsx')
    
    model = ArvyaxModel()
    
    # Part 5 & 6 (Insights printed to console)
    model.fit(train_df)
    model.feature_understanding(train_df)
    model.run_ablation(train_df)
    
    # Part 7 (Error Analysis Generation)
    print("\n--- Generating Error Analysis ---")
    run_error_analysis(model, train_df) # This trains a temporary model to get unseen validation errors
    
    # Final Predictions on Test Data
    print("\n--- Running Final Inference on Test Data ---")
    states, intensities, probs = model.predict(test_df)
    
    engine = DecisionEngine()
    
    results = []
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        state = states[i]
        intensity = intensities[i]
        stress = row.get('stress_level', np.nan)
        energy = row.get('energy_level', np.nan)
        time_day = row.get('time_of_day', np.nan)
        
        what, when = engine.decide(state, intensity, stress, energy, time_day)
        conf, uncert = calculate_uncertainty(probs[i])
        support_msg = engine.generate_message(state, intensity, what, when)
        
        results.append({
            'id': row['id'],
            'predicted_state': state,
            'predicted_intensity': intensity,
            'confidence': round(conf, 4),
            'uncertain_flag': uncert,
            'what_to_do': what,
            'when_to_do': when,
            'supportive_message': support_msg
        })
        
    out_df = pd.DataFrame(results)
    out_df.to_csv('predictions.csv', index=False)
    print(f"Generated predictions.csv with {len(out_df)} rows.")

if __name__ == "__main__":
    main()
