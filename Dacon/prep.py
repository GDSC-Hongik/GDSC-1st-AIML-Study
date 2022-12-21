import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

from utils import get_values


class Prep:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        self.train_df = train_df
        self.val_df = val_df

    def _imputation(self) -> pd.DataFrame:
        tr_df = self.train_df.copy()
        val_df = self.val_df.copy()

        tr_df["암의 장경"] = tr_df["암의 장경"].fillna(tr_df["암의 장경"].mean())
        tr_df = tr_df.fillna(0)

        val_df["암의 장경"] = val_df["암의 장경"].fillna(val_df["암의 장경"].mean())
        val_df = val_df.fillna(0)

        return tr_df, val_df

    def _scaling(
        self, imputed_tr: pd.DataFrame, imputed_val: pd.DataFrame
    ) -> pd.DataFrame:
        numeric_cols = [
            "나이",
            "암의 장경",
            "ER_Allred_score",
            "PR_Allred_score",
            "KI-67_LI_percent",
            "HER2_SISH_ratio",
        ]
        ignore_cols = ["ID", "img_path", "mask_path", "수술연월일", "N_category"]

        for col in imputed_tr.columns:
            if col in ignore_cols:
                continue

            if col in numeric_cols:
                scaler = StandardScaler()
                imputed_tr[col] = scaler.fit_transform(get_values(imputed_tr[col]))
                imputed_val[col] = scaler.transform(get_values(imputed_val[col]))

            else:
                le = LabelEncoder()
                imputed_tr[col] = le.fit_transform(get_values(imputed_tr[col]))
                imputed_val[col] = le.transform(get_values(imputed_val[col]))

        return imputed_tr, imputed_val

    def run(
        self,
    ):
        imputed_tr, imputed_val = self._imputation()
        scaled_tr, scaled_val = self._scaling(
            imputed_tr=imputed_tr, imputed_val=imputed_val
        )

        return scaled_tr, scaled_val
