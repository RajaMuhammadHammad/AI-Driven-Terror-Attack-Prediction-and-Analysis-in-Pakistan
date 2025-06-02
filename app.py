from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import folium  # for map
import pandas as pd
import pickle
import json
import google.generativeai as genai
import os

app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load ML models and encoders
with open("models/decision_tree_model.pkl", "rb") as f:
    dt_model = pickle.load(f)
with open("models/dt_columns.pkl", "rb") as f:
    dt_columns = pickle.load(f)

with open("models/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)
with open("models/knn_model_columns.pkl", "rb") as f:
    knn_columns = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load terrorism dataset (optional, no longer used by chatbot)
# terrorism_df = pd.read_csv("Finalize_processed_pakistanterrorattacks.csv")

def load_accuracies():
    try:
        with open('models/accuracies.json') as f:
            return json.load(f)
    except:
        return {
            "decision_tree": {"train": 0, "test": 0},
            "knn": {"train": 0, "test": 0}
        }


@app.route("/")
def home():
    accuracies = load_accuracies()
    return render_template("index.html",
        dt_train_accuracy=accuracies.get("decision_tree", {}).get("train", 0),
        dt_test_accuracy=accuracies.get("decision_tree", {}).get("test", 0),
        knn_train_accuracy=accuracies.get("knn", {}).get("train", 0),
        knn_test_accuracy=accuracies.get("knn", {}).get("test", 0)
    )

@app.route("/decision_tree")
def decision_tree_page():
    return render_template("decision_tree.html")

@app.route("/knn")
def knn_page():
    return render_template("knn.html")

@app.route("/predict_decision_tree", methods=["POST"])
def predict_decision_tree():
    input_data = {
        "Year": int(request.form["Year"]),
        "Month": int(request.form["Month"]),
        "Province": request.form["Province"],
        "Weapon_type": request.form["Weapon_type"],
        "Group": request.form["Group"]
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    for col in dt_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[dt_columns]

    prediction_numeric = dt_model.predict(df)[0]
    prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

    return render_template("decision_tree.html", prediction_result=prediction_label)

@app.route("/predict_knn", methods=["POST"])
def predict_knn():
    input_data = {
        "Year": int(request.form["Year"]),
        "Month": int(request.form["Month"]),
        "Province": request.form["Province"],
        "Weapon_type": request.form["Weapon_type"],
        "Group": request.form["Group"]
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    for col in knn_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[knn_columns]

    prediction = knn_model.predict(df)
    prediction_label = prediction[0]

    return render_template("knn.html", prediction_result=prediction_label)

# === Updated Chatbot powered by Gemini only (no dataset search) ===
model = genai.GenerativeModel('gemini-2.0-flash')

@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_page():
    response = None
    if request.method == "POST":
        user_question = request.form.get("user_question", "").strip()

        # Gemini-compatible prompt (no system role)
        prompt = (
    "Answer ONLY questions related to terrorism activity in Pakistan, including predictions for 2025 based on historical data.\n"
    "If the user asks something unrelated, reply with:\n"
    "<p><strong>Sorry, I can only answer questions related to terrorism activity in Pakistan up to 2024 and predictions for 2025.</strong></p>\n"
    "Avoid saying things like 'I'm ready to answer'. Just respond directly.\n"
    "Use <strong>, <ul>, <br> tags in your answer if helpful.\n\n"
    f"User's question: {user_question}"
        )


        try:
            result = model.generate_content(prompt)
            response = result.text
        except Exception as e:
            response = f"<p>Error generating response: {e}</p>"

    return render_template("chatbot.html", response=response)


@app.route("/clusters")
def clusters():
    try:
        # Load the terrorism data
        df = pd.read_csv("Finalize_processed_pakistanterrorattacks.csv")

        # Filter valid rows (drop NaNs)
        df = df.dropna(subset=["latitude", "longitude", "Casualties"])

        # Select features for clustering
        features = df[["latitude", "longitude", "Casualties"]]

        # Apply KMeans
        kmeans = KMeans(n_clusters=4, random_state=42)
        df["Cluster"] = kmeans.fit_predict(features)

        # Create Folium map centered on Pakistan
        map_center = [30.3753, 69.3451]
        m = folium.Map(location=map_center, zoom_start=5)

        # Color palette for clusters
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

        # Add points to map
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=colors[row["Cluster"] % len(colors)],
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(f"""
                    <strong>City:</strong> {row['city']}<br>
                    <strong>Province:</strong> {row['Province']}<br>
                    <strong>Casualties:</strong> {int(row['Casualties'])}<br>
                    <strong>Group:</strong> {row['Group']}<br>
                    <strong>Attack Type:</strong> {row['AttackType']}
                """, max_width=250)
            ).add_to(m)

        # Save map to HTML file
        m.save("templates/cluster_map.html")
        return render_template("cluster_map.html")

    except Exception as e:
        return f"<p>Error generating cluster map: {e}</p>"


@app.route("/kmeans_elbow")
def kmeans_elbow():
    try:
        df = pd.read_csv("Finalize_processed_pakistanterrorattacks.csv")
        features = ['latitude', 'longitude', 'Killed', 'Wounded', 'Casualties']
        df_cluster = df[features].fillna(0)

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_cluster)

        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Make directories if not exist
        elbow_dir = "static/plots/elbow"
        cluster_dir = "static/plots/cluster"
        os.makedirs(elbow_dir, exist_ok=True)
        os.makedirs(cluster_dir, exist_ok=True)

        elbow_path = os.path.join(elbow_dir, "elbow_plot.png")
        cluster_path = os.path.join(cluster_dir, "cluster_plot.png")

        # Elbow Plot
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS (Inertia)')
        plt.grid(True)
        plt.savefig(elbow_path)
        plt.close()

        # Cluster Plot with K=3
        optimal_k = 3
        kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        df['Cluster'] = cluster_labels

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='longitude', y='latitude', hue='Cluster', palette='viridis')
        plt.title(f'KMeans Clustering of Terrorist Attacks (K={optimal_k})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(cluster_path)
        plt.close()

        return render_template("kmeans_elbow.html",
                               elbow_img=elbow_path,
                               cluster_img=cluster_path)

    except Exception as e:
        return f"<p>Error generating plots: {e}</p>"


if __name__ == "__main__":
    app.run(debug=True)
