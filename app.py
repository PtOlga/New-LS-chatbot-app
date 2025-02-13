def build_knowledge_base():
    import time
    
    start_time = time.time()
    documents = []
    st.info("Starting knowledge base creation...")
    
    # Create progress bar
    progress_bar = st.progress(0)
    total_urls = len(urls)
    
    for idx, url in enumerate(urls):
        try:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
            st.write(f"✅ Loaded content from {url}")
            # Update progress bar
            progress_bar.progress((idx + 1) / total_urls)
        except (RequestException, Timeout) as e:
            st.write(f"❌ Error loading page {url}: {e}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Show chunks info
    st.write(f"📄 Split into {len(chunks)} chunks")
    
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    vector_store.save_local(VECTOR_STORE_PATH)
    
    # Calculate metrics
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Calculate size of vector store directory
    total_size = 0
    for path, dirs, files in os.walk(VECTOR_STORE_PATH):
        for f in files:
            fp = os.path.join(path, f)
            total_size += os.path.getsize(fp)
    
    size_mb = total_size / (1024 * 1024)
    
    # Display completion message
    st.success(f"""
    ✨ Knowledge base creation completed:
    ⏱️ Time taken: {time_taken:.2f} seconds
    💾 Size: {size_mb:.2f} MB
    🔢 Total chunks: {len(chunks)}
    """)
    
    return vector_store