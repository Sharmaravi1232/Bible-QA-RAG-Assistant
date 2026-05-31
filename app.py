import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# ----------------------------
# EMBEDDINGS
# ----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------
# LOAD FAISS
# ----------------------------

db = FAISS.load_local(
    ".",
    embeddings,
    index_name="index",
    allow_dangerous_deserialization=True
)

# ----------------------------
# LOAD MODEL
# ----------------------------

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

# ----------------------------
# SAFETY FILTER
# ----------------------------

def moderate_query(query):

    blocked = [
        "christian terrorism",
        "religious violence",
        "hate muslims",
        "kill unbelievers",
        "religious supremacy"
    ]

    q = query.lower()

    for item in blocked:
        if item in q:
            return False

    return True

# ----------------------------
# QUERY EXPANSION
# ----------------------------

def expand_query(question):

    mapping = {
        "who was moses":
        "Moses prophet Israel Exodus Sinai",

        "who was jesus":
        "Jesus Christ Messiah Son of God",

        "forgiveness":
        "forgiveness mercy repentance sin",

        "salvation":
        "saved eternal life salvation",

        "faith":
        "faith trust God belief",

        "love":
        "love God neighbor charity"
    }

    q = question.lower()

    for key in mapping:
        if key in q:
            return mapping[key]

    return question

# ----------------------------
# RETRIEVE CONTEXT
# ----------------------------

def retrieve_context(question):

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15
        }
    )

    docs = retriever.invoke(
        expand_query(question)
    )

    context = []

    for doc in docs:

        ref = doc.metadata.get(
            "reference",
            "Bible Reference"
        )

        context.append(
            f"{ref}: {doc.page_content}"
        )

    return "\n\n".join(context)

# ----------------------------
# PROMPT
# ----------------------------

def build_prompt(question, denomination, context):

    return f"""
You are a Christianity-focused AI Assistant.

Denomination:
{denomination}

Rules:

- Use supplied scripture.
- Cite references.
- Stay biblical.
- Never invent Bible verses.

Scripture:

{context}

Question:
{question}

Answer:
"""

# ----------------------------
# MAIN QA FUNCTION
# ----------------------------

def bible_qa(question, denomination):

    if not question:
        return "Please enter a question."

    if not moderate_query(question):
        return "I cannot assist with hateful or harmful religious content."

    context = retrieve_context(question)

    prompt = build_prompt(
        question,
        denomination,
        context
    )

    response = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    answer = response[0]["generated_text"]

    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1]

    return answer

# ----------------------------
# PRAYER
# ----------------------------

def generate_prayer(topic):

    prompt = f"""
Write a Christian prayer about:

{topic}
"""

    response = generator(
        prompt,
        max_new_tokens=150
    )

    return response[0]["generated_text"]

# ----------------------------
# SERMON
# ----------------------------

def generate_sermon(topic):

    prompt = f"""
Write a Christian sermon about:

{topic}
"""

    response = generator(
        prompt,
        max_new_tokens=250
    )

    return response[0]["generated_text"]

# ----------------------------
# DEVOTIONAL
# ----------------------------

def generate_devotional(topic):

    prompt = f"""
Write a Christian devotional about:

{topic}
"""

    response = generator(
        prompt,
        max_new_tokens=250
    )

    return response[0]["generated_text"]

# ----------------------------
# UI
# ----------------------------

with gr.Blocks(title="BibleGuard AI") as demo:

    gr.Markdown("# ✝️ BibleGuard AI")

    with gr.Tab("Bible QA"):

        denomination = gr.Dropdown(
            [
                "Catholic",
                "Protestant",
                "Orthodox",
                "Non-Denominational"
            ],
            value="Non-Denominational",
            label="Denomination"
        )

        question = gr.Textbox(
            label="Ask a Bible Question"
        )

        answer = gr.Textbox(
            lines=10,
            label="Answer"
        )

        ask_btn = gr.Button("Ask")

        ask_btn.click(
            bible_qa,
            inputs=[
                question,
                denomination
            ],
            outputs=answer
        )

    with gr.Tab("Prayer"):

        prayer_topic = gr.Textbox(
            label="Prayer Topic"
        )

        prayer_output = gr.Textbox(
            lines=10
        )

        prayer_btn = gr.Button(
            "Generate Prayer"
        )

        prayer_btn.click(
            generate_prayer,
            inputs=prayer_topic,
            outputs=prayer_output
        )

    with gr.Tab("Sermon"):

        sermon_topic = gr.Textbox(
            label="Sermon Topic"
        )

        sermon_output = gr.Textbox(
            lines=12
        )

        sermon_btn = gr.Button(
            "Generate Sermon"
        )

        sermon_btn.click(
            generate_sermon,
            inputs=sermon_topic,
            outputs=sermon_output
        )

    with gr.Tab("Devotional"):

        devotional_topic = gr.Textbox(
            label="Devotional Topic"
        )

        devotional_output = gr.Textbox(
            lines=12
        )

        devotional_btn = gr.Button(
            "Generate Devotional"
        )

        devotional_btn.click(
            generate_devotional,
            inputs=devotional_topic,
            outputs=devotional_output
        )

demo.launch(server_name="0.0.0.0")
