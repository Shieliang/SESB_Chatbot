import streamlit as st
import boto3
import os
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ================= 配置区 =================
BUCKET_NAME = 'sagemaker-us-east-1-987762561422' # 你的桶名
DOC_PREFIX = 'Documents/'
FORM_PREFIX = 'Forms/'
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
INDEX_PATH = "./faiss_index_cache"

# ================= 页面配置 =================
st.set_page_config(page_title="SESB 智能客服", page_icon="⚡")
# === 顶部标题栏 (带清空按钮) ===
col1, col2 = st.columns([5, 1])
with col1:
    st.title("⚡ SESB 智能业务助手")
with col2:
    if st.button("🗑️ 清空", key="reset_btn_top", use_container_width=True):
        # 1. 重置 UI：保留欢迎语
        st.session_state.messages = [{"role": "assistant", "content": "您好，我是 SESB 智能客服。请问有什么可以帮您？"}]
        
        # 2. 销毁 AI 记忆 (强制重置)
        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]
        if "memory" in st.session_state:
            del st.session_state["memory"]
        
        # 3. 刷新页面
        st.rerun()

# ================= 资源初始化 (带缓存) =================
@st.cache_resource
def init_resources():
    s3 = boto3.client('s3')
    
    # 获取表格列表
    available_forms = []
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=FORM_PREFIX)
        if 'Contents' in response:
            for obj in response['Contents']:
                fname = os.path.basename(obj['Key'])
                if fname.lower().endswith('.pdf'): available_forms.append(fname)
    except Exception as e: st.error(f"S3 连接错误: {e}")
    
    # 加载向量库
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")
    if os.path.exists(INDEX_PATH):
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # 如果没有缓存，则生成 (Streamlit 会转圈圈提示用户)
        if not os.path.exists('/tmp/docs'): os.makedirs('/tmp/docs')
        all_docs = []
        resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=DOC_PREFIX)
        if 'Contents' in resp:
            for obj in resp['Contents']:
                if obj['Key'].endswith('.pdf'):
                    path = f"/tmp/docs/{os.path.basename(obj['Key'])}"
                    s3.download_file(BUCKET_NAME, obj['Key'], path)
                    all_docs.extend(PyPDFLoader(path).load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_documents(all_docs)
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(INDEX_PATH)
    
    return s3, available_forms, vectorstore

s3, available_forms, vectorstore = init_resources()
forms_str = ", ".join(available_forms) if available_forms else "无"

# ================= 链条初始化 =================
# 使用 session_state 保证链条在对话中持久存在
if "qa_chain" not in st.session_state:
    llm = ChatBedrock(model_id=MODEL_ID, model_kwargs={"max_tokens": 1000})
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key="question", output_key="answer"
    )
    
    sesb_template = f"""
    你是一名 SESB (Sabah Electricity Sdn Bhd) 的专业客服。
    你的服务范围**仅限于**：电力申请、账单查询、电表相关、停电故障、承包商信息及 SESB 相关政策。
    
    <rules>
    1. 【业务边界 - 关键！】
       - 如果用户的问题与 SESB 电力业务**无关**（例如：询问水费、天气、政治、数学题、其他公司业务、闲聊等）：
       - **必须** 拒绝回答。
       - 标准回复话术：“抱歉，我是 SESB 电力客服，无法回答与电力服务无关的问题。请问有什么关于电表或账单的事宜我可以帮您吗？”
    
    2. 【资料来源】
       - 必须基于【参考资料】回答。如果资料里没有答案，就说“目前的资料里没有相关信息”，不要编造。
    
    3. 【隐私例外】
       - 资料里的 Contractor (承包商) 电话/地址是公开信息，**必须直接提供**。
    
    4. 【表格下载】
       - 推荐列表：[{forms_str}]。告诉用户“可以下载 [文件名]”。
    
    5. 【身份界限】
       - 你是客服，我是用户。不要重复我的问题，不要自言自语。
    </rules>
    
    【对话历史】：
    {{chat_history}}
    
    【参考资料】：
    {{context}}
    
    用户问题：{{question}}
    
    请直接回答：
    """
    
    SESB_PROMPT = PromptTemplate(template=sesb_template, input_variables=["context", "question", "chat_history"])
    
    condense_template = """
    任务：将后续问题改写为一个独立的、完整的问题。
    <rules>
    1. 如果用户问“我说过什么”或“我住在哪里”，请务必查看 <chat_history> 并将具体信息补充进问题里。
    2. 保持语言与用户输入一致（如果用户用华语，就用华语改写；如果用马来语，就用马来语）。
    3. 不要回答问题，只需输出改写后的问题。
    </rules>
    聊天历史: {chat_history}
    后续输入: {question}
    独立问题:"""
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory,
        return_source_documents=True, condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": SESB_PROMPT}
    )

# ================= 聊天界面逻辑 =================

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "您好，我是 SESB 智能客服。请问有什么可以帮您？"}]

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True) # 允许 HTML 渲染下载按钮

# 处理用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用 AI
    with st.chat_message("assistant"):
        with st.spinner("正在查询 SESB 资料库..."):
            try:
                res = st.session_state.qa_chain.invoke({"question": prompt})
                answer = res['answer']
                
                # 检查表格并生成按钮 HTML
                found_forms = list(set([f for f in available_forms if f.replace('.pdf','').replace('.PDF','').lower() in answer.lower()]))
                if found_forms:
                    answer += "<br><br>📂 <b>推荐下载：</b><br>"
                    for fname in found_forms:
                        try:
                            link = s3.generate_presigned_url('get_object', Params={'Bucket': BUCKET_NAME, 'Key': f"{FORM_PREFIX}{fname}"}, ExpiresIn=3600)
                            # 使用 HTML 渲染漂亮的按钮
                            answer += f"""<a href="{link}" target="_blank" style="background-color:#0073bb;color:white;padding:5px 10px;text-decoration:none;border-radius:15px;margin:2px;display:inline-block;">⬇️ {fname}</a> """
                        except: pass
                
                st.markdown(answer, unsafe_allow_html=True)
                
                # 存入历史
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"系统错误: {e}")