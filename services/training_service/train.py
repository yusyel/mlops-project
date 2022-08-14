# # 1. Data Data Preparation
#%%
import pandas as pd
import numpy as np

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
import mlflow
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("mlops-project")
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
client.list_experiments()

@task(name="read_data")
def read(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.education = df.education.fillna(0)
    df.bpmeds = df.bpmeds.fillna(0)
    df.cigsperday = df.cigsperday.fillna(df.cigsperday.mean())
    df.heartrate = df.heartrate.fillna(df.heartrate.mean())
    df.glucose = df.glucose.fillna(df.glucose.mean())
    df.totchol = df.totchol.fillna(df.totchol.mean())
    df.bmi = df.bmi.fillna(df.bmi.mean())
    return df


@task(name="split_dataframe")
def split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    #return len(df_train), len(df_val), len(df_test)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = df_train.tenyearchd.values
    y_val = df_val.tenyearchd.values
    y_test = df_test.tenyearchd.values
    del df_train["tenyearchd"]
    del df_val["tenyearchd"]
    del df_test["tenyearchd"]
    return df_train, df_val, y_train, y_val


@task(name="prepare_dicts")
def dicts(df_train, df_val, y_train, y_val):
    train_dicts = df_train.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X_val = dv.fit_transform(val_dicts)
    return train_dicts, X_train, val_dicts, X_val

@task(name="train_models")
def train_models(train_dicts, y_train, val_dicts, y_val):
    logger = get_run_logger()
    def objective(space):
        with mlflow.start_run():
            mlflow.set_tag("mlops", "model1")
            pipeline = make_pipeline(
                DictVectorizer(sparse=False),
                RandomForestClassifier(
                    criterion=space["criterion"],
                    max_depth=space["max_depth"],
                    max_features=space["max_features"],
                    min_samples_leaf=space["min_samples_leaf"],
                    min_samples_split=space["min_samples_split"],
                    n_estimators=space["n_estimators"],
                    max_leaf_nodes=space["max_leaf_nodes"],
                    n_jobs=-1,
                ),
            )
            mlflow.log_params(space)
            pipeline.fit(train_dicts, y_train)
            y_pred_val = pipeline.predict_proba(val_dicts)[:, 1]
            auc = roc_auc_score(y_val, y_pred_val)
            acc = accuracy_score(y_val, y_pred_val >= 0.55)
            f1 = f1_score(y_val, y_pred_val >= 0.5, average="weighted")
            metrics = {"accuracy_score": acc, "auc": auc, "f1": f1}
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
        return {"loss": auc, "status": STATUS_OK}

    space = {
        "criterion": hp.choice("criterion", ["entropy", "gini"]),
        "max_depth": scope.int(hp.quniform("max_depth", 1, 2000, 1)),
        "max_features": hp.choice("max_features", ["auto", "sqrt", "log2", None]),
        "min_samples_leaf": hp.uniform("min_samples_leaf", 0, 0.5),
        "min_samples_split": hp.uniform("min_samples_split", 0, 1),
        "n_estimators": scope.int(hp.quniform("n_estimators", 1, 2000, 1)),
        "max_leaf_nodes": scope.int(hp.quniform("max_leaf_nodes", 2, 1000, 1)),
        "class_weight": hp.choice("class_weight", ["balanced", "balanced_subsample", None]),
        "ccp_alpha": hp.uniform("ccp_alpha", 0, 1),
    }

    rstate = np.random.default_rng(2)
    params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        rstate=rstate,
        trials=Trials(),
    )

    best = client.search_runs(
        experiment_ids="1",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.auc DESC"],
    )[0]

    best_model = best.info.run_id
    logger.info(f"best model run id is:{best_model}")
    return best_model

@flow(task_runner=SequentialTaskRunner())
def register():
    logger = get_run_logger()
    df = read(path="./framingham.csv")
    df_train, df_val, y_train, y_val = split(df)
    train_dicts, X_train, val_dicts, X_val  = dicts(df_train, df_val, y_train, y_val)
    best_model = train_models(train_dicts, y_train, val_dicts, y_val)

    model =    mlflow.register_model(model_uri=f"runs:/{best_model}/model", name="chd_risk_model")
    client.transition_model_version_stage(
        name="chd_risk_model",
        version=1,
        stage="Production"
    )
    logger.info(f"Training is done! {model}")

    return model
register()
