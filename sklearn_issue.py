import shap
from sklearn.model_selection import train_test_split

from econml.solutions.causal_analysis import CausalAnalysis


def main():
    print("Fetching data")
    # Fetch the data
    X, y = shap.datasets.adult()
    target_feature = "income"
    y = [1 if y_i else 0 for y_i in y]

    full_data = X.copy()
    full_data[target_feature] = y

    data_train, data_test = train_test_split(
        full_data,
        test_size=1000,
        random_state=96132,
        stratify=full_data[target_feature],
    )

    categorical_columns = [
        "Race",
        "Sex",
        "Workclass",
        "Marital Status",
        "Country",
        "Occupation",
    ]

    target_feature = "income"

    # Create the analysis object
    print("Creating analysis object")
    analysis = CausalAnalysis(
        feature_inds=["Age", "Sex"],
        categorical=categorical_columns,
        heterogeneity_inds=["Marital Status"],
        classification=True,
        nuisance_models="automl",
        heterogeneity_model="forest",
        upper_bound_on_cat_expansion=49,
        skip_cat_limit_checks=False,
        n_jobs=1,
        categories="auto",
        verbose=True,
        random_state=100,
    )


if __name__ == "__main__":
    main()
