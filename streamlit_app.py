import traceback

try:
    from bulkrna.app import main
    main()
except Exception as exc:
    try:
        import streamlit as st
        st.error("BulkRNA Explorer encountered an unexpected error.")
        st.exception(exc)
        with st.expander("Technical traceback"):
            st.code(traceback.format_exc())
    except Exception:
        raise
