"""Optional shared-password gate for hosted WebUI deployments.

Set APP_PASSWORD to require it; leave unset for local use (no prompt).
"""

import hmac
import os


def require_password() -> None:
    import streamlit as st

    expected = os.getenv("APP_PASSWORD", "")
    if not expected or st.session_state.get("_slidesme_authed"):
        return
    st.title("🔒 slidesme")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), expected.encode()):
            st.session_state["_slidesme_authed"] = True
            st.rerun()
        st.error("Wrong password")
    st.stop()
