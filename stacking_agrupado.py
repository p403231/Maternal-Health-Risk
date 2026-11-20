import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.validation import check_is_fitted


class StackingAgrupado(ClassifierMixin, BaseEstimator):

    def __init__(self, estimators, final_estimator=None,
                 n_splits=5, random_state=42):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y):

        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.classes_ = np.unique(y)

        grupos = pd.util.hash_pandas_object(
            X, index=False
        )

        cv_interno = StratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        particiones = list(
            cv_interno.split(X, y, groups=grupos)
        )

        meta_features = []
        self.estimators_ = []

        for nombre, estimador in self.estimators:

            pred_oof = np.zeros(
                (len(X), len(self.classes_))
            )

            for idx_train, idx_val in particiones:

                modelo = clone(estimador)

                modelo.fit(
                    X.iloc[idx_train],
                    y.iloc[idx_train]
                )

                probabilidades = modelo.predict_proba(
                    X.iloc[idx_val]
                )

        
                for j, clase in enumerate(modelo.classes_):
                    pos = np.where(self.classes_ == clase)[0][0]
                    pred_oof[idx_val, pos] = probabilidades[:, j]

            meta_features.append(pred_oof)

            
            modelo_final = clone(estimador)
            modelo_final.fit(X, y)

            self.estimators_.append(
                (nombre, modelo_final)
            )

        X_meta = np.hstack(meta_features)

        if self.final_estimator is None:
            meta_modelo = LogisticRegression(max_iter=1000)
        else:
            meta_modelo = clone(self.final_estimator)

        self.final_estimator_ = meta_modelo
        self.final_estimator_.fit(X_meta, y)

        return self

    def predict_proba(self, X):

        check_is_fitted(self, "final_estimator_")

        meta_features = []

        for nombre, modelo in self.estimators_:

            probabilidades = modelo.predict_proba(X)

            pred_alineadas = np.zeros(
                (len(X), len(self.classes_))
            )

            for j, clase in enumerate(modelo.classes_):
                pos = np.where(self.classes_ == clase)[0][0]
                pred_alineadas[:, pos] = probabilidades[:, j]

            meta_features.append(pred_alineadas)

        X_meta = np.hstack(meta_features)

        return self.final_estimator_.predict_proba(X_meta)

    def predict(self, X):

        check_is_fitted(self, "final_estimator_")

        meta_features = [
            modelo.predict_proba(X)
            for _, modelo in self.estimators_
        ]

        X_meta = np.hstack(meta_features)

        return self.final_estimator_.predict(X_meta)
