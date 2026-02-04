from retrieval.semantic_search import semantic_search
from retrieval.web_search import web_search
from llm.answer_llm import answer_with_llm

SCORE_THRESHOLD = 1.2   # 🔴 RELAXED for TF-IDF

def answer_query(query):
    results = semantic_search(query)

    print("\n🔎 Internal search results (score):")
    for doc, score in results:
        print(f"Score: {score:.3f} | Text preview: {doc.page_content[:80]}")

    good_docs = []
    for doc, score in results:
        if score <= SCORE_THRESHOLD:
            good_docs.append(doc.page_content)

    if good_docs:
        print("✅ Answered from INTERNAL documents")
        return "\n\n".join(good_docs)

    print("🌐 Falling back to WEB search")
    #web_context = web_search(query)
    #return answer_with_llm(web_context, query)
    return "No relevant internal information found."

