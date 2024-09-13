import os
from uuid import uuid4

from flask import Flask, redirect, render_template, request, session

from main import process_question

app = Flask(__name__)

app.secret_key = "grigrjogrjo"


UPLOAD_FOLDER = "uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


MAX_CONTEXT_LENGTH = 5  # Number of previous exchanges to keep


def get_conversation_history():
    if "conversation_history" not in session:
        session["conversation_history"] = []
    return session["conversation_history"]


def add_to_conversation_history(question, answer):
    history = get_conversation_history()
    history.append({"question": question, "answer": answer})
    if len(history) > MAX_CONTEXT_LENGTH:
        history.pop(0)
    session["conversation_history"] = history


@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if "user_id" not in session:
        session["user_id"] = str(uuid4())

    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)

            session["uploaded_file_path"] = file_path

        if "uploaded_file_path" not in session:
            return redirect(request.url)

        file_path = session["uploaded_file_path"]

        question = request.form["question"]
        conversation_history = get_conversation_history()
        result = process_question(question, conversation_history, file_path)

        add_to_conversation_history(question, result)

    return render_template(
        "index.html",
        gmaps_api_key=os.getenv("GPLACES_API_KEY"),
        result=result,
    )


if __name__ == "__main__":
    app.run(debug=True)
