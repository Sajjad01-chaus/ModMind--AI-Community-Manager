import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq


def ingest_knowledge(urls: list[str], files: list) -> FAISS:
    documents = []
    # URLs 
    if urls:
        try:
            loader = UnstructuredURLLoader(urls=urls)
            docs = loader.load()
            if any(d.page_content.strip() for d in docs):
                documents.extend(docs)
            else:
                raise ValueError
        except Exception:
            from langchain_community.document_loaders import SeleniumURLLoader
            loader = SeleniumURLLoader(urls=urls)
            documents.extend(loader.load())

    # files
    for file in files or []:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getvalue())
        if file.name.endswith('.pdf'):
            loader = PyPDFLoader(path)
        elif file.name.endswith('.txt'):
            loader = TextLoader(path)
        else:
            loader = Docx2txtLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore

# Query agent

def get_answer(
    vectorstore: FAISS,
    brand_name: str,
    brand_voice: str,
    question: str,
    model_name: str,
    temperature: float,
    k: int
) -> str:
    llm = ChatGroq(model_name=model_name, temperature=temperature, streaming=False)

    system_template = """
You are the official AI Community Moderator for {brand_name}, speaking in a friendly, natural tone.

Brand Voice Guidelines for your responses:
{brand_voice}

Let Zara is the brand name and following are the sample questions and responses for you to give as Community Moderator.
##Sample Questions:
Q: Why is Zara famous?  
A: Zara is famous for pioneering fast fashion—turning runway trends into stores in just weeks. [Source: Document: zara_about.pdf]

Q: What are Zara’s main product lines?  
A:
- Women’s ready-to-wear [Source: URL https://zara.com/women]  
- Men’s casual and formal [Source: URL https://zara.com/men]  
- Kids & baby apparel [Source: Document: kids_catalog.pdf]

**(Real-Time Product Query)**  
Q: What is the average price of a men’s leather jacket, and is it in stock?  
A:  
- Average price: \$120–\$160 per piece. [Source: URL https://zara.com/men/leather-jackets]  
- Availability: Most styles are in stock online; sizes M and L ship immediately, XL ships in 2–3 days. [Source: URL https://zara.com/men/leather-jackets/stock]


Intstructions:
- Answer the question based on the context provided.
- Use only the provided context. If the answer isn’t there, say “I’m sorry, I don’t have that information.”  
- Cite every fact inline in square brackets (e.g. “[Source: URL https://…]” or “[Source: Document: name.pdf]”).  
- Keep responses to **≤ 3 sentences** or a single short paragraph if question is expecting much information.  
- Use contractions (“I’m”, “you’re”), short sentences, and a friendly tone.  
- For lists or multi-step answers, use bullet points—no more than 5 items.



CONTEXT:
{context}

QUESTION:
{input}
"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    result = chain.invoke({
        "input": question,
        "brand_name": brand_name or "the brand",
        "brand_voice": brand_voice or ""
    })
    return result.get("answer", "")