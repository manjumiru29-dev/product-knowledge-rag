import streamlit as st
import requests
import json
import os
from datetime import datetime
import pandas as pd

# ─── CONFIG ────────────────────────────────────────────────
RAG_BACKEND_URL = "http://localhost:8002"
RESULTS_FILE    = "chatbot_results.json"

SOURCE_TYPES = [
    None,
    "product",
    "article",
    "faq",
    "brand",
    "category",
]


# ─── BACKEND HELPERS ───────────────────────────────────────

def check_backend_health() -> bool:
    try:
        r = requests.get(
            f"{RAG_BACKEND_URL}/health", timeout=5
        )
        return r.status_code == 200
    except Exception:
        return False


def query_rag(
    question: str,
    source_type: str = None,
    top_k: int = 5,
) -> dict:
    """Send question to RAG backend and get answer."""
    try:
        payload = {"question": question}
        if source_type:
            payload["source_type"] = source_type
        if top_k:
            payload["top_k"] = top_k

        r = requests.post(
            f"{RAG_BACKEND_URL}/api/query",
            json    = payload,
            timeout = 300,
        )
        if r.status_code == 200:
            return r.json()
        return {
            "answer": f"⚠️ Backend returned status {r.status_code}",
            "retrieved_documents": [],
            "context": "",
        }
    except requests.exceptions.ConnectionError:
        return {
            "answer": (
                "❌ Cannot connect to RAG backend. "
                "Please make sure FastAPI is running on port 8002."
            ),
            "retrieved_documents": [],
            "context": "",
        }
    except Exception as e:
        return {
            "answer": f"❌ Error: {str(e)}",
            "retrieved_documents": [],
            "context": "",
        }


def fetch_all_items(
    source_type: str,
    top_k: int = 100,
) -> list:
    try:
        r = requests.get(
            f"{RAG_BACKEND_URL}/api/list",
            params  = {
                "source_type": source_type,
                "top_k":       top_k,
            },
            timeout = 30,
        )
        if r.status_code == 200:
            data = r.json()
            return data.get("items", [])
        return []
    except Exception:
        return []


# ─── INTENT DETECTION ──────────────────────────────────────

def detect_listing_type(question: str):
    q = question.lower()

    if any(k in q for k in [
        "brand", "brands", "all brand",
        "list brand", "show brand",
        "brand list", "brand name",
        "which brand", "available brand",
    ]):
        return "brand", "Brands"

    if any(k in q for k in [
        "categor", "categories", "all categor",
        "list categor", "show categor",
        "category list", "product category",
        "which category", "available category",
    ]):
        return "category", "Categories"

    if any(k in q for k in [
        "faq", "faqs", "all faq",
        "list faq", "show faq",
        "frequently asked", "common question",
    ]):
        return "faq", "FAQs"

    if any(k in q for k in [
        "article", "articles", "all article",
        "list article", "show article",
        "guide", "guides", "all guide",
        "tutorial", "tutorials",
    ]):
        return "article", "Articles"

    if any(k in q for k in [
        "product", "products", "all product",
        "list product", "show product",
        "show all", "list all", "give me all",
        "display all", "get all", "fetch all",
        "tell me all", "show me all",
        "list the all", "available product",
        "what product", "which product",
    ]):
        return "product", "Products"

    return None, None


def is_listing_query(question: str) -> bool:
    q = question.lower()
    listing_kw = [
        "list", "show all", "show me all",
        "all brands", "all products", "all categories",
        "all items", "all faq", "all articles",
        "all guides", "all tutorials",
        "what brands", "what products", "what categories",
        "give me all", "display all", "get all",
        "fetch all", "tell me all", "list the",
        "list all", "show the all", "list the all",
        "brand list", "category list", "product list",
        "available brand", "available product",
        "which brand", "which category",
    ]
    return any(k in q for k in listing_kw)


# ─── TABLE BUILDERS ────────────────────────────────────────

def build_products_table(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "No":           len(rows) + 1,
            "Product Name": item.get("name", ""),
            "SKU":          item.get("sku", ""),
            "Category":     item.get("category", ""),
            "Price (₹)":    item.get("price", 0),
            "Discount (%)": item.get("discount_percentage", 0),
            "Status":       item.get("status", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_categories_table(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "No":          len(rows) + 1,
            "Icon":        item.get("icon", ""),
            "Category":    item.get("name", ""),
            "Slug":        item.get("slug", ""),
            "Description": item.get("description", "")[:80]
                           if item.get("description") else "",
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_articles_table(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "No":         len(rows) + 1,
            "Title":      item.get("name", ""),
            "Type":       item.get("article_type", ""),
            "Difficulty": item.get("difficulty_level", ""),
            "Read Time":  f"{item.get('read_time', 0)} min",
            "Category":   item.get("category", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_faqs_table(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        answer = item.get("description", "") or ""
        rows.append({
            "No":       len(rows) + 1,
            "Question": item.get("name", ""),
            "Answer":   answer[:120] + "..."
                        if len(answer) > 120 else answer,
            "Category": item.get("category", ""),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_brands_table(items: list) -> pd.DataFrame:
    rows = []
    for item in items:
        rows.append({
            "No":             len(rows) + 1,
            "Brand Name":     item.get("name", ""),
            "Products Count": item.get("product_count", 0),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_listing_table(items: list, list_type: str) -> pd.DataFrame:
    builders = {
        "Products":   build_products_table,
        "Categories": build_categories_table,
        "Articles":   build_articles_table,
        "FAQs":       build_faqs_table,
        "Brands":     build_brands_table,
    }
    builder = builders.get(list_type)
    if builder:
        return builder(items)
    return pd.DataFrame()


# ─── RAG CARD HELPERS ──────────────────────────────────────

def parse_doc_fields(doc: dict) -> dict:
    """
    Extract clean fields from a retrieved document.
    Parses 'content' string like:
      "Product: X. Brand: Y. Status: available. Category: Z. Description: ..."
    Falls back to top-level doc keys if not found.
    """
    content = doc.get("content", "")
    fields = {}

    # Primary: split on ". " (dot-space) — matches backend format
    for part in content.split(". "):
        if ": " in part:
            k, _, v = part.partition(": ")
            fields[k.strip()] = v.strip()

    # Fallback: split on " - " if too few fields parsed
    if len(fields) < 3:
        for part in content.split(" - "):
            if ": " in part:
                k, _, v = part.partition(": ")
                if k.strip() not in fields:
                    fields[k.strip()] = v.strip()

    def get(*keys, fallback="N/A"):
        for k in keys:
            for fk, fv in fields.items():
                if fk.lower() == k.lower():
                    return fv
            v = doc.get(k)
            if v not in (None, "", 0):
                return str(v)
        return fallback

    # Extract ONLY the description text after "Description:" key
    raw_description = "N/A"
    if "Description:" in content:
        raw_description = content.split("Description:")[-1].strip().strip(". ")
    else:
        raw_description = get("Description", "description", "summary", fallback="N/A")

    # Category: show full value + Parent Category if available
    category_raw = get("Category", "category", fallback="N/A")
    parent_cat   = get("Parent Category", "parent_category", fallback="")
    if parent_cat and parent_cat != category_raw and parent_cat != "N/A":
        category_display = f"{category_raw} (Parent: {parent_cat})"
    else:
        category_display = category_raw

    # article_type only exists in articles table, not products
    source       = doc.get("source_type", "")
    article_type = get("article_type", "Article Type", "Article", fallback="")
    if article_type and article_type not in ("N/A", ""):
        article_display = article_type
    elif source == "article":
        article_display = get("type", fallback="N/A")
    else:
        article_display = "N/A (Product)"

    return {
        "product_name": get(
            "Product", "Product Name", "name", "title", "course_title",
            fallback=doc.get("course_title", "Unknown Product"),
        ),
        "sku":          get("SKU", "sku"),
        "price":        get("Price", "Rs", "price", "price_inr"),
        "discount":     get("Discount", "discount_percentage", "discount"),
        "status":       get("Status", "status", "availability"),
        "brand":        get("Brand", "brand", "brand_name"),
        "category":     category_display,
        "source_type":  source or "product",
        "article":      article_display,
        "difficulty":   get("difficulty_level", "Difficulty", fallback="N/A"),
        "read_time":    get("read_time", "Read Time", fallback="N/A"),
        "description":  raw_description,
        "similarity":   doc.get("similarity", 0),
    }


def display_product_card(i: int, doc: dict):
    """Render one retrieved document as a structured product card."""
    f   = parse_doc_fields(doc)
    sim = float(f["similarity"])

    if sim >= 0.75:
        badge_color = "#22c55e"
    elif sim >= 0.55:
        badge_color = "#f59e0b"
    else:
        badge_color = "#ef4444"

    rows = [
        ("<b> Product Name</b>", f["product_name"]),
        ("<b> SKU</b>",          f["sku"]),
        ("<b> Price</b>",         f["price"]),
        ("<b> Discount</b>",      f["discount"]),
        ("<b> Status</b>",         f["status"]),
        ("<b> Brand</b>",          f["brand"]),
        ("<b> Category</b>",       f["category"]),
        ("<b> Article Type</b>",   f["article"]),
        ("<b> Description</b>",    f["description"]),
    ]

    table_rows_html = ""
    for label, value in rows:
        table_rows_html += (
            "<tr>"
            "<td style='padding:5px 0;color:#64748b;width:150px;vertical-align:top'>"
            + label +
            "</td>"
            "<td style='padding:5px 0;color:#1e293b;line-height:1.6'>"
            + str(value) +
            "</td>"
            "</tr>"
        )

    card_html = (
        "<div style='background:#f8fafc;border:1px solid #e2e8f0;"
        "border-left:4px solid " + badge_color + ";"
        "border-radius:8px;padding:16px 20px;margin-bottom:12px;'>"
        "<div style='font-size:17px;font-weight:700;color:#1e293b;margin-bottom:12px;'>"
        + str(i) + ". " + f["product_name"] +
        "<span style='font-size:12px;font-weight:500;"
        "background:" + badge_color + "22;color:" + badge_color + ";"
        "padding:2px 8px;border-radius:12px;margin-left:8px;'>"
        + f"{sim:.0%} match" +
        "</span></div>"
        "<table style='width:100%;border-collapse:collapse;font-size:14px;'>"
        + table_rows_html +
        "</table></div>"
    )

    st.markdown(card_html, unsafe_allow_html=True)


# ─── RESULT STORAGE ────────────────────────────────────────

def save_result(
    question, answer, source_type,
    retrieved_count, action_type="rag",
):
    result = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question":        question,
        "answer":          answer,
        "source_type":     source_type or "All",
        "retrieved_count": retrieved_count,
        "answer_length":   len(answer),
        "action_type":     action_type,
    }
    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    results.append(result)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ─── REPORT PAGE ───────────────────────────────────────────

def display_report():
    results = load_results()
    if not results:
        st.warning("No results yet. Ask some questions first!")
        return

    df = pd.DataFrame(results)
    st.markdown("##  Results Report")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries",     len(df))
    c2.metric("Avg Answer Length", f"{df['answer_length'].mean():.0f}")
    c3.metric("Avg Items Found",   f"{df['retrieved_count'].mean():.1f}")
    c4.metric("Unique Questions",  df["question"].nunique())

    st.divider()

    tab1, tab2 = st.tabs([" Summary Table", "💬 Individual Responses"])

    with tab1:
        cols = [
            "timestamp", "question", "source_type",
            "retrieved_count", "answer_length", "action_type",
        ]
        display_df = df[[c for c in cols if c in df.columns]].copy()
        display_df.columns = [
            "Timestamp", "Question", "Source Type",
            "Items Retrieved", "Answer Length", "Action",
        ][:len(display_df.columns)]
        st.dataframe(display_df, use_container_width=True)

        if "action_type" in df.columns:
            st.markdown("### Action Type Distribution")
            st.bar_chart(df["action_type"].value_counts())

    with tab2:
        for idx, row in df.iterrows():
            with st.expander(f"Q{idx+1}: {row['question'][:60]}..."):
                st.markdown(f"** Time:** {row['timestamp']}")
                st.markdown(f"** Question:** {row['question']}")
                st.markdown(f"** Source:** {row['source_type']}")
                st.markdown(f"** Action:** {row.get('action_type','rag').upper()}")
                st.markdown(f"** Items:** {row['retrieved_count']}")
                st.markdown("** Answer:**")
                st.info(row["answer"])

    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        label     = "⬇ Download Results as CSV",
        data      = csv,
        file_name = f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime      = "text/csv",
    )


# ─── PAGE SETUP ────────────────────────────────────────────

st.set_page_config(
    page_title            = "Product Knowledge Chatbot",
    page_icon             = "🛍️",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ─── SIDEBAR ───────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🛍️ Product Chatbot")
    st.markdown("*Powered by RAG + flan-t5-base*")
    st.divider()

    st.markdown("### 🔌 Backend Status")
    if check_backend_health():
        st.success("✅ RAG Backend Connected")
        st.caption("FastAPI running on port 8002")
    else:
        st.error("❌ Backend Not Running")
        st.caption("Start FastAPI on port 8002")

    st.divider()

    st.markdown("### 🔍 Search Filter")
    source_type = st.selectbox(
        "Filter by content type:",
        options = SOURCE_TYPES,
        format_func = lambda x: (
            "🌐 All Content" if x is None else {
                "product":  " Products",
                "article":  " Articles",
                "faq":      " FAQs",
                "brand":    " Brands",
                "category": " Categories",
            }.get(x, x.title())
        ),
    )

    st.divider()

    st.markdown("###  Settings")
    top_k = st.slider(
        "Max results to retrieve:",
        min_value = 1,
        max_value = 10,
        value     = 5,
    )

    st.divider()

    st.markdown("###  Current Settings")
    st.write(" Model: **flan-t5-base**")
    st.write(f"🔍 Filter: **{source_type or 'All Content'}**")
    st.write(f" Top-K: **{top_k}**")
    st.write("🌐 Backend: **localhost:8002**")

    st.divider()

    st.markdown("###  Results")
    if st.button(" View Report", use_container_width=True):
        st.session_state.show_report = True
    if st.button(" Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    if st.button(" Clear Results", use_container_width=True):
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
            st.success("Results cleared!")
            st.rerun()

    st.divider()

    st.markdown("###  Quick Lists")
    if st.button(" All Brands", use_container_width=True):
        st.session_state.starter = "list all brands"
        st.rerun()
    if st.button("All Categories", use_container_width=True):
        st.session_state.starter = "list all categories"
        st.rerun()
    if st.button(" All Products", use_container_width=True):
        st.session_state.starter = "list all products"
        st.rerun()
    if st.button(" All FAQs", use_container_width=True):
        st.session_state.starter = "list all faq"
        st.rerun()
    if st.button(" All Articles", use_container_width=True):
        st.session_state.starter = "list all articles"
        st.rerun()


# ─── MAIN AREA ─────────────────────────────────────────────

if st.session_state.get("show_report", False):
    if st.button("← Back to Chat"):
        st.session_state.show_report = False
        st.rerun()
    display_report()

else:
    st.title("🛍️ Product Knowledge Chatbot")
    st.markdown(
        "Ask me anything about our products! "
        "I search the product knowledge base "
        "and give you accurate answers."
    )

    if not check_backend_health():
        st.error(
            " RAG Backend is not running!\n\n"
            "```uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload```"
        )

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Welcome buttons
    if not st.session_state.messages:
        st.markdown("### 💬 Start a conversation")

        st.markdown("**Product Questions:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(" What products do you have?", use_container_width=True):
                st.session_state.starter = "What products do you have?"
                st.rerun()
        with col2:
            if st.button(" Show me best products", use_container_width=True):
                st.session_state.starter = "Show me your best products"
                st.rerun()
        with col3:
            if st.button(" Show product categories", use_container_width=True):
                st.session_state.starter = "What are your product categories?"
                st.rerun()

        st.markdown("** Browse All Items:**")
        col4, col5, col6, col7 = st.columns(4)
        with col4:
            if st.button(" List all brands", use_container_width=True):
                st.session_state.starter = "list all brands"
                st.rerun()
        with col5:
            if st.button(" List all categories", use_container_width=True):
                st.session_state.starter = "list all categories"
                st.rerun()
        with col6:
            if st.button(" List all products", use_container_width=True):
                st.session_state.starter = "list all products"
                st.rerun()
        with col7:
            if st.button(" List all FAQs", use_container_width=True):
                st.session_state.starter = "list all faq"
                st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show retrieved docs (history replay)
            if (
                message["role"] == "assistant"
                and message.get("retrieved_docs")
            ):
                docs = message["retrieved_docs"]
                with st.expander(f"🔍 View {len(docs)} retrieved item(s)"):
                    for i, doc in enumerate(docs, 1):
                        display_product_card(i, doc)

            # Show listing table
            if (
                message["role"] == "assistant"
                and message.get("listing_df") is not None
            ):
                df_show = message["listing_df"]
                if not df_show.empty:
                    st.dataframe(df_show, use_container_width=True)

    starter_prompt = st.session_state.pop("starter", None)

    user_prompt = (
        st.chat_input("Ask a product question or list brands/categories...")
        or starter_prompt
    )

    if user_prompt:

        with st.chat_message("user"):
            st.markdown(user_prompt)

        st.session_state.messages.append({
            "role":    "user",
            "content": user_prompt,
        })

        list_type, list_display = detect_listing_type(user_prompt)
        listing = is_listing_query(user_prompt)

        # ══════════════════════════════════════════════════
        # PATH 1 — LISTING QUERY
        # ══════════════════════════════════════════════════
        if listing and list_type:

            with st.spinner(f" Fetching {list_display} from database..."):
                items = fetch_all_items(source_type=list_type, top_k=100)

            if not items:
                answer     = (
                    f" No {list_display} found.\n\n"
                    f"Make sure the `/api/list` endpoint "
                    f"is added to your `endpoints.py`."
                )
                listing_df = pd.DataFrame()
            else:
                listing_df = build_listing_table(items, list_display)
                count      = len(listing_df)
                answer     = (
                    f"###  {list_display} List\n\n"
                    f"Found **{count} {list_display}** in the database:"
                )

            with st.chat_message("assistant"):
                st.markdown(answer)
                if not listing_df.empty:
                    st.dataframe(listing_df, use_container_width=True)

            st.session_state.messages.append({
                "role":           "assistant",
                "content":        answer,
                "retrieved_docs": [],
                "listing_df":     listing_df,
            })
            save_result(
                question        = user_prompt,
                answer          = answer,
                source_type     = list_type,
                retrieved_count = len(items),
                action_type     = "listing",
            )

        # ══════════════════════════════════════════════════
        # PATH 2 — RAG QUERY
        # ══════════════════════════════════════════════════
        else:
            with st.spinner("🔍 Searching product knowledge base..."):
                result    = query_rag(user_prompt, source_type, top_k)
                answer    = result.get("answer", "No answer found.")
                retrieved = result.get("retrieved_documents", [])

            best_score = max(
                (d.get("similarity", 0) for d in retrieved),
                default=0,
            )

            if best_score < 0.5 or not retrieved:
                answer = (
                    f" I could not find relevant product information for "
                    f"**'{user_prompt}'**.\n\n"
                    f"Please ask about our products, brands, categories, or FAQs.\n\n"
                    f" **Tip:** Use the **Quick Lists** buttons in the sidebar "
                    f"to browse all items."
                )

            with st.chat_message("assistant"):
                st.markdown(answer)

                if 0 < best_score < 0.5:
                    st.warning(f" Low similarity score ({best_score:.2f})")

                if retrieved and best_score >= 0.5:
                    with st.expander(
                        f"🔍 View {len(retrieved)} retrieved product(s)"
                    ):
                        for i, doc in enumerate(retrieved, 1):
                            display_product_card(i, doc)

                elif not retrieved:
                    st.caption("ℹ No matching products found.")

            st.session_state.messages.append({
                "role":           "assistant",
                "content":        answer,
                "retrieved_docs": retrieved,
                "listing_df":     None,
            })
            save_result(
                question        = user_prompt,
                answer          = answer,
                source_type     = source_type,
                retrieved_count = len(retrieved),
                action_type     = "rag",
            )