from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

app = Flask(__name__)

# Data Historis
data = {
    "Hari": list(range(1, 21)),
    "Suhu": [30.1, 30.5, 31.0, 31.2, 31.5, 32.0, 32.3, 32.8, 33.0, 33.5,
             33.8, 34.0, 34.5, 34.9, 35.2, 35.5, 36.0, 36.5, 36.8, 37.0]
}
df = pd.DataFrame(data)

# Model
X = df[["Hari"]]
y = df["Suhu"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Fungsi untuk simpan grafik prediksi dinamis
def simpan_grafik_dinamis(hari_input=None, prediksi_input=None):
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Hari"], df["Suhu"], color="orange", label="Data Historis")
    plt.plot(df["Hari"], model.predict(df[["Hari"]]), color="green", label="Model Regresi")

    if hari_input is not None and prediksi_input is not None:
        plt.scatter(hari_input, prediksi_input, color="red", s=100, label="Prediksi Input")

    plt.xlabel("Hari")
    plt.ylabel("Suhu (°C)")
    plt.title("Prediksi Suhu Berdasarkan Hari")
    plt.legend()
    plt.tight_layout()

    # Pastikan folder static ada
    os.makedirs("static", exist_ok=True)
    plt.savefig("static/prediksi.png")
    plt.close()

# Buat grafik awal
simpan_grafik_dinamis()

@app.route("/", methods=["GET", "POST"])
def index():
    prediksi = None
    hari = None
    if request.method == "POST":
        hari = int(request.form["hari"])
        prediksi = model.predict(np.array([[hari]]))[0]
        simpan_grafik_dinamis(hari, prediksi)

    return render_template("index.html", prediksi=prediksi, hari=hari,
                           mae=mae, mse=mse, r2=r2)

if __name__ == "__main__":
    app.run(debug=True)
