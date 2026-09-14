import traceback

try:
    from bulkrna.app import main
    main()

    # Explicit navigation link so Custom GSEA is always visible even if the
    # automatic multipage menu is collapsed or easy to miss.
    import streamlit as st
    st.sidebar.divider()
    st.sidebar.page_link("pages/5_Custom_GSEA.py", label="🧬 Custom GSEA")
except Exception as exc:
    try:
        import streamlit as st
        st.error("BulkRNA Explorer encountered an unexpected error.")
        st.exception(exc)
        with st.expander("Technical traceback"):
            st.code(traceback.format_exc())
    except Exception:
        raise
