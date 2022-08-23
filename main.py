from flask import Flask, render_template, request, url_for
import os
import sys
import logging
import pandas as pd
from static.python.module import *
import json

app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# maximum file size 16 megabytes
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000
#UPLOAD_FOLDER = './static/uploads/'
#ALLOWED_EXTENSIONS = {'csv', 'txt'}
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.static_folder = 'static'


@app.route("/", methods=["GET", "POST"])
def get_data():
    if request.method == "POST":
        my_data = [request.form["ulasan"]]
        df = pd.DataFrame(my_data, columns=['review'])
        df["review"] = df.apply(identify_tokens, axis=1)
        seq_text = get_encode(df['review'])

        # making prediction of categories and sentiment
        pred_result, percentage_cat, percentage_sent = get_prediction_cat_sentiment(
            seq_text)

        print(percentage_cat)
        print(percentage_sent)
        print(pred_result)

        percentage_cat = percentage_cat.tolist()
        percentage_sent = percentage_sent.tolist()
        return render_template(
            "result.html",
            pred_result=pred_result,
            percentage_cat=percentage_cat[0],
            percentage_sent=percentage_sent[0],
            review=my_data,
            show=True)

        # return redirect(request.url, water_data=my_data)
    return render_template("tool.html")


@app.route("/about")
def about():
    return render_template("about-us.html")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.3", port=5003, threaded=True)
