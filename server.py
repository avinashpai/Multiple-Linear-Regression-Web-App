"""
By: Avinash Pai

Flask web application that implements multiple linear regression.
In its current state:
    - Allows for users to upload data as CSV files
    - Displays given data in a table
    - Ability to choose features and output variable
    - Displays scatter plot for each feature against output
    - Prompts users for test-train split percentage
    - Displays actual vs. predicited histogram and important metrics

TODO:
 - Reorganize/restructure codebase
 - Pretty up the html with css
 - Add input field for predicitions on new data
 - Implement regularized multi linear regression (Ridge, Lasso, etc.)
 - Add functionality for uploading training and test data sets as an option
    instead of splitting.
 - Much more!
"""

import os
import ast
import base64
import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from io import BytesIO

UPLOAD_FOLDER = "/Users/avinashpai/Documents/flask_model/uploads"
ALLOWED_EXTENSIONS = {"csv"}
TWO_DIGIT_NUM = re.compile("^[0-9]{1,2}[:.,-]?$")
NUM_LIST = re.compile(r"^\d+(?:[ \t]*,[ \t]*\d+)+$")


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = b"mw\x149Y\xc8\xa8\xe0o0\xdb\xe3\xa5\xef J"

data = pd.DataFrame()
test_size = 0


def train(features, dep_var, test_size):
    global data
    # Compute linear regression
    X = data[features].values
    y = data[dep_var].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0
    )

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)  # training the algorithm

    # coeff_df = pd.DataFrame(regressor.coef_, X.columns,
    # columns=["Coefficient"])

    y_pred = regressor.predict(X_test)
    model_eq = create_eq(regressor)

    act_vs_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    fig = Figure()
    ax = fig.subplots()
    ax.grid(which="major", linestyle="-", linewidth="0.5", color="green")
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="black")
    act_vs_pred.plot(kind="bar", ax=ax, figsize=(10, 8))
    # possibly switch to scatter/line for actual vs interpolation

    buf = BytesIO()
    fig.savefig(buf, format="png")
    model_fig = base64.b64encode(buf.getbuffer()).decode("ascii")

    metrics_dict = {
        "mae": metrics.mean_absolute_error(y_test, y_pred),
        "mse": metrics.mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        "r2": metrics.r2_score(y_test, y_pred),
        "model_eq": model_eq,
    }

    return model_fig, metrics_dict, regressor


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def create_eq(model):
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    model_eq = f"y = {model.intercept_:.3f}"
    for i in range(len(model.coef_)):
        model_eq += f" + {model.coef_[i]:.3f}x{str(i+1).translate(SUB)}"

    return model_eq


@app.route("/", methods=["GET", "POST"])
def upload_file():
    global data
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

            data = pd.read_csv(f"./uploads/{filename}")
            data.columns = [
                column.replace(" ", "_").lower() for column in data
            ]

            return redirect(url_for("table", filename=filename))
    return render_template("upload.html")


# only linear regression for now
@app.route("/table/<filename>", methods=["GET", "POST"])
def table(filename):
    # read csv into pandas DataFrame
    global data

    if request.method == "POST":
        dep_var = request.form["dep_var_opt"]
        features = request.form.getlist("features")

        return redirect(
            url_for(
                "model", filename=filename, features=features, dep_var=dep_var
            )
        )

    # page that prompts users with different features to plot
    # NOTES:
    # use the html templating to create a page and render this along with the
    # prompt to users to pick features to plot
    return render_template(
        "table.html",
        filename=filename,
        tables=[data.to_html(classes="data", header="true")],
        option_list=list(data.columns),
    )


@app.route("/model/<filename>/<features>vs<dep_var>", methods=["GET", "POST"])
def model(filename, features, dep_var):
    global data
    global test_size

    fig_data = []

    features = ast.literal_eval(features)
    features = [n.strip() for n in features]

    y = data[dep_var]

    sns.set()
    if request.method == "POST":

        if "train_test_split" in request.form and re.match(
            TWO_DIGIT_NUM, request.form["train_test_split"]
        ):
            test_size = 1 - (int(request.form["train_test_split"]) / 100)

            model_fig, metrics_dict, regressor = train(
                features, dep_var, test_size
            )

            return render_template(
                "plot.html",
                model_fig=model_fig,
                metrics=metrics_dict,
                filename=filename,
            )
        elif "test_features" in request.form and re.match(
            NUM_LIST, request.form["test_features"]
        ):
            test_features_list = [
                float(feature)
                for feature in request.form["test_features"].split(",")
            ]
            test_features = np.asarray(test_features_list)
            model_fig, metrics_dict, regressor = train(
                features, dep_var, test_size
            )
            pred = regressor.predict(test_features.reshape(1, -1))

            return render_template(
                "plot.html",
                model_fig=model_fig,
                metrics=metrics_dict,
                filename=filename,
                pred=pred,
            )

        else:
            flash("Invalid Input!")
            return redirect(request.url)

    for feature in features:
        X = data[feature]
        fig = Figure()
        ax = fig.subplots()
        ax.scatter(X, y)
        ax.set_title(f"{feature} vs {dep_var}")
        ax.set_xlabel(feature)
        ax.set_ylabel(dep_var)

        buf = BytesIO()
        fig.savefig(buf, format="png")
        fig_data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))

    return render_template("plot.html", fig_data=fig_data, filename=filename)
