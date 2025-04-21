from lazyllm import OnlineEmbeddingModule, Document, Retriever, OnlineChatModule, ChatPrompter

prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')
document = Document(dataset_path="rag_master", embed=OnlineEmbeddingModule().start(), manager=False)
retriever = Retriever(document, Document.CoarseChunk, similarity="cosine", topk=3, output_format='content')
m = OnlineChatModule(stream=True).prompt(ChatPrompter(prompt, extra_keys=["context"]))
query = input('enter your query\n')
r = m(dict(query=query, context=''.join(retriever(query))))
print(r)
