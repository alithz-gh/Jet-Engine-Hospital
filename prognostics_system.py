"""
PrognosticsSystem: loads the exported artifact bundle and reproduces notebook
inference exactly, for one engine's history up to a chosen cycle. This is the
same class used by the notebook (for final evaluation) and the Streamlit app
(for interactive inference) -- there is exactly one inference code path.
"""
import numpy as np
import pandas as pd
import pickle


class PrognosticsSystem:
    def __init__(self, bundle_path='artifacts/prognostics_bundle.pkl'):
        with open(bundle_path, 'rb') as f:
            b = pickle.load(f)
        self.pipe = b['feature_pipeline']
        self.window_cols = b['window_cols']
        self.current_cols = b['current_cols']
        self.rul_model = b['rul_model']
        self.rul_cap = b['rul_cap']
        self.conformal_margin = b['conformal_margin_90']
        self.clf_models = b['clf_models']
        self.anomaly_scaler = b['anomaly_scaler']
        self.anomaly_models = b['anomaly_models']
        self.anomaly_refs = b['anomaly_refs']
        self.anomaly_threshold = b['anomaly_threshold']
        self.persistence = b['persistence']
        self.cost_weights = b['cost_weights']
        self.metadata = b['metadata']

    def compute_features(self, engine_history_raw):
        """engine_history_raw: raw sensor dataframe for ONE engine, all cycles up to 'now'."""
        return self.pipe.transform_engine(engine_history_raw)

    def predict_rul(self, features_row, interval=True):
        X = features_row[self.window_cols].values.reshape(1, -1)
        pred = float(np.clip(self.rul_model.predict(X)[0], 0, None))
        if not interval:
            return {'point': pred}
        lower = max(0, pred - self.conformal_margin)
        upper = pred + self.conformal_margin
        return {'point': pred, 'lower': lower, 'upper': upper, 'width': upper - lower}

    def _calibrate_probability(self, calibrator, raw_prob):
        eps = 1e-6
        logit = np.log((raw_prob + eps) / (1 - raw_prob + eps))
        X = np.array([[logit]])

        try:
            return float(calibrator.predict_proba(X)[0, 1])
        except AttributeError:
            # Fall back to a manual logistic transform if pickled model is missing newer sklearn internals.
            if hasattr(calibrator, 'coef_') and hasattr(calibrator, 'intercept_'):
                coef = np.asarray(calibrator.coef_)
                intercept = np.asarray(calibrator.intercept_)
                if coef.ndim == 1:
                    score = float(X.dot(coef) + intercept)
                elif coef.shape[0] == 1:
                    score = float(X.dot(coef.T) + intercept[0])
                elif coef.shape[0] == 2 and intercept.shape[0] == 2:
                    score = float(X.dot((coef[1] - coef[0]).T) + (intercept[1] - intercept[0]))
                else:
                    raise
                return float(1 / (1 + np.exp(-score)))
            raise

    def failure_risk(self, features_row, horizons=(10, 20, 30)):
        X = features_row[self.window_cols].values.reshape(1, -1)
        out = {}
        for h in horizons:
            m = self.clf_models[h]
            raw_prob = float(m['model'].predict_proba(X)[0, 1])
            if m['calibrator'] is not None:
                prob = self._calibrate_probability(m['calibrator'], raw_prob)
                calibrated = True
            else:
                prob = raw_prob
                calibrated = False
            out[h] = {'probability': prob, 'threshold': m['threshold'],
                       'alert': prob >= m['threshold'], 'calibrated': calibrated}
        return out

    def anomaly_score(self, features_row):
        X_raw = features_row[self.current_cols].values.reshape(1, -1)
        X_scaled = self.anomaly_scaler.transform(X_raw)

        raw_scores = {
            'IsolationForest': -self.anomaly_models['IsolationForest'].score_samples(X_scaled)[0],
            'LOF': -self.anomaly_models['LOF'].score_samples(X_scaled)[0],
            'OneClassSVM': -self.anomaly_models['OneClassSVM'].decision_function(X_scaled)[0],
        }
        percentiles = {}
        for name, raw in raw_scores.items():
            ref_sorted = np.sort(self.anomaly_refs[name])
            pct = np.searchsorted(ref_sorted, raw, side='right') / len(ref_sorted) * 100
            percentiles[name] = float(pct)
        combined_pct = float(np.mean(list(percentiles.values())))
        margin = combined_pct - self.anomaly_threshold
        return {'percentiles': percentiles, 'combined_percentile': combined_pct,
                'margin': margin, 'alert': combined_pct >= self.anomaly_threshold}

    def _persistent(self, alert_history, m=None, n=None):
        m = m or self.persistence['m']
        n = n or self.persistence['n']
        recent = alert_history[-n:]
        return sum(recent) >= m

    def recommend(self, rul_out, risk_out, anomaly_out, clf_h10_history, clf_h20_history,
                  clf_h30_history, anomaly_alert_history):
        """
        clf_h*_history / anomaly_alert_history: lists of booleans (alert fired or not)
        for the engine's cycles up to and including 'now', used to evaluate persistence.
        """
        persistent_h10 = self._persistent(clf_h10_history)
        persistent_h20 = self._persistent(clf_h20_history)
        persistent_h30 = self._persistent(clf_h30_history)
        persistent_anomaly = self._persistent(anomaly_alert_history)

        reasons = []
        level = 'CONTINUE'

        # STOP triggers
        stop_from_rul = rul_out['lower'] <= 10
        stop_from_h10 = persistent_h10
        if stop_from_rul or stop_from_h10:
            level = 'STOP'
            if stop_from_rul:
                reasons.append(f"RUL lower bound ({rul_out['lower']:.0f} cycles) at or below safety floor")
            if stop_from_h10:
                reasons.append("persistent high-confidence failure-within-10-cycles alert")

            # Uncertainty override: RUL-only trigger with a wide interval -> downgrade
            if stop_from_rul and not stop_from_h10 and rul_out['width'] > 50:
                level = 'INSPECT'
                reasons = [f"RUL point estimate is low but interval is wide "
                           f"(±{rul_out['width']/2:.0f} cycles) -- not confident enough for STOP"]

        # INSPECT triggers (only checked if not already STOP)
        if level != 'STOP':
            if persistent_h20:
                level = 'INSPECT'; reasons.append("persistent failure-within-20-cycles alert")
            if persistent_h30:
                level = 'INSPECT'; reasons.append("persistent failure-within-30-cycles alert")
            if persistent_anomaly:
                level = 'INSPECT'; reasons.append("persistent anomaly detection alert")
            if rul_out['width'] > 40 and rul_out['point'] < 60:
                level = 'INSPECT'; reasons.append("wide RUL uncertainty in an actionable range")

        # Disagreement flag: anomaly says something, everything else says healthy
        disagreement = persistent_anomaly and not (persistent_h10 or persistent_h20 or persistent_h30) \
                        and rul_out['lower'] > 30
        if disagreement and level == 'CONTINUE':
            level = 'INSPECT'
            reasons.append("signals disagree: anomaly detector flags deviation while RUL/classifier "
                            "signals show healthy status -- shown rather than silently resolved")

        if not reasons:
            reasons.append("all signals within normal range")

        return {'level': level, 'reasons': reasons, 'disagreement': disagreement,
                 'persistent_h10': persistent_h10, 'persistent_h20': persistent_h20,
                 'persistent_h30': persistent_h30, 'persistent_anomaly': persistent_anomaly}
