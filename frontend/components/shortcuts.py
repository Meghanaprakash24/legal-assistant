"""
components/shortcuts.py
========================
Keyboard shortcuts for the Indian Legal AI Assistant.

Streamlit has no native keyboard-shortcut API, so this module injects a
small ``<script>`` block that listens for the relevant key combinations
and clicks a target Streamlit button by its visible label. This means
the *real* logic for each action still lives in a normal
``st.button(...)`` on the page — this module only simulates the click.

Usage on a page:

    from components.shortcuts import inject_keyboard_shortcuts

    send_clicked = st.button("Send Message", key="send_btn")
    clear_clicked = st.button("Clear Chat", key="clear_btn")
    refresh_clicked = st.button("Refresh Dashboard", key="refresh_btn")

    inject_keyboard_shortcuts(
        send_button_label="Send Message",
        clear_button_label="Clear Chat",
        refresh_button_label="Refresh Dashboard",
    )

Only pass the labels for buttons that actually exist on the current
page — omitted shortcuts are simply not wired up.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def inject_keyboard_shortcuts(
    *,
    send_button_label: Optional[str] = None,
    clear_button_label: Optional[str] = None,
    refresh_button_label: Optional[str] = None,
    show_hint: bool = True,
) -> None:
    """Wire Ctrl+Enter / Ctrl+L / Ctrl+R to existing Streamlit buttons.

    Args:
        send_button_label: Visible label of the "send message" button on
            this page (e.g. "Send Message"). Bound to Ctrl+Enter.
        clear_button_label: Visible label of the "clear chat" button.
            Bound to Ctrl+L.
        refresh_button_label: Visible label of the "refresh" button.
            Bound to Ctrl+R.
        show_hint: Whether to render a small caption listing the active
            shortcuts for discoverability.
    """
    bindings: list[tuple[str, str]] = []
    if send_button_label:
        bindings.append(("Enter", send_button_label))  # Ctrl+Enter handled via ctrlKey check below
    if clear_button_label:
        bindings.append(("l", clear_button_label))
    if refresh_button_label:
        bindings.append(("r", refresh_button_label))

    if not bindings:
        return

    # Build a small JS map of key -> button label, matched against
    # Streamlit's rendered <button> text content at keydown time (rather
    # than baking in a DOM query at injection time, since Streamlit
    # re-renders buttons on every script run).
    js_bindings = ", ".join(f'"{key.lower()}": {label!r}' for key, label in bindings)

    st.markdown(
        f"""
        <script>
        (function() {{
            const bindings = {{{js_bindings}}};
            if (window.__legalAiShortcutsBound) return;
            window.__legalAiShortcutsBound = true;

            document.addEventListener("keydown", function(event) {{
                if (!event.ctrlKey && !event.metaKey) return;
                const key = event.key.toLowerCase();
                const label = bindings[key];
                if (!label) return;

                const buttons = window.parent.document.querySelectorAll("button");
                for (const btn of buttons) {{
                    if (btn.innerText && btn.innerText.trim().includes(label)) {{
                        event.preventDefault();
                        btn.click();
                        break;
                    }}
                }}
            }});
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )

    if show_hint:
        hint_parts = []
        if send_button_label:
            hint_parts.append("`Ctrl + Enter` Send")
        if clear_button_label:
            hint_parts.append("`Ctrl + L` Clear Chat")
        if refresh_button_label:
            hint_parts.append("`Ctrl + R` Refresh")
        st.caption(" · ".join(hint_parts))