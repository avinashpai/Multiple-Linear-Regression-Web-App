import os
import ast
import base64
import numpy as np
import pandas as pd
import seaborn as sns
import json
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    send_from_directory,
    render_template,
)
from werkzeug.utils import secure_filename
from io import BytesIO

UPLOAD_FOLDER = "/Users/avinashpai/Documents/flask_model/uploads"
ALLOWED_EXTENSIONS = {"csv"}


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

data = pd.DataFrame()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rerurn(filename):
    return redirect(url_for("table", filename=filename))


@app.route("/", methods=["GET", "POST"])
def upload_file():
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
            return redirect(url_for("table", filename=filename))
    return render_template("upload.html")


# only linear regression for now
@app.route("/table/<filename>", methods=["GET", "POST"])
def table(filename):
    # read csv into pandas DataFrame
    global data
    data = pd.read_csv(f"./uploads/{filename}")
    data.columns = [x.lower().capitalize() for x in data.columns]

    if request.method == "POST":
        dep_var = request.form["dep_var_opt"]
        features = request.form.getlist("features")

        return redirect(
            url_for("model", filename=filename, features=features, dep_var=dep_var)
        )

    # page that prompts users with different features to plot
    # NOTES:
    # use the html templating to create a page and render this along with the prompt to users to pick features to plot
    return render_template(
        "table.html",
        filename=filename,
        tables=[data.to_html(classes="data", header="true")],
        option_list=list(data.columns),
    )


@app.route("/model/<filename>/<features>vs<dep_var>", methods=["GET", "POST"])
def model(filename, features, dep_var):
    global data

    fig_data = []

    features = ast.literal_eval(features)
    features = [n.strip() for n in features]

    y = data[dep_var]

    sns.set()
    if request.method == "POST":

        if "rerun" in request.form:
            return redirect(url_for("table", filename=filename))

        # Compute linear regression
        X = data[features].values
        y = data[dep_var].values

        test_size = 1 - (int(request.form["train_test_split"]) / 100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)  # training the algorithm

        # coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=["Coefficient"])

        y_pred = regressor.predict(X_test)

        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        model_eq = f"y = {regressor.intercept_:.3f}"
        for i in range(len(regressor.coef_)):
            model_eq += f" + {regressor.coef_[i]:.3f}x{str(i+1).translate(SUB)}"

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

        return render_template(
            "plot.html", model_fig=model_fig, metrics=metrics_dict, filename=filename
        )

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

