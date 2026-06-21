"""
components/exports.py
======================
Export/download helpers for the Indian Legal AI Assistant.

Every function renders one or more ``st.download_button`` widgets and
returns nothing — callers just call the relevant ``*_export_buttons``
function wherever they want the buttons to appear. No file is written
to disk; everything is generated in-memory and streamed via Streamlit's
download button.

PDF export uses a minimal, dependency-light approach (plain-text PDF via
the ``fpdf2`` library if installed) and falls back to a clear message
directing the user to the Markdown/TXT export if ``fpdf2`` isn't
available, rather than silently failing.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Iterable

import streamlit as st


# =============================================================================
# Chat export — PDF / Markdown / TXT
# =============================================================================


def _chat_to_markdown(messages: Iterable[dict[str, Any]], title: str = "Legal Assistant Conversation") -> str:
    """Render a chat message list as a Markdown transcript.

    Args:
        messages: Iterable of dicts with at least ``role`` and
            ``content`` keys (the standard Streamlit chat message shape).
        title: Document title.
    """
    lines = [f"# {title}", "", f"_Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_", ""]
    for msg in messages:
        role = str(msg.get("role", "unknown")).capitalize()
        content = str(msg.get("content", ""))
        lines.append(f"### {role}")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def _chat_to_txt(messages: Iterable[dict[str, Any]]) -> str:
    """Render a chat message list as plain text."""
    lines = []
    for msg in messages:
        role = str(msg.get("role", "unknown")).upper()
        content = str(msg.get("content", ""))
        lines.append(f"[{role}]")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)


def _chat_to_pdf_bytes(messages: Iterable[dict[str, Any]], title: str = "Legal Assistant Conversation") -> bytes | None:
    """Render a chat message list as a simple PDF, if ``fpdf2`` is installed.

    Returns:
        PDF bytes, or ``None`` if ``fpdf2`` is not available.
    """
    try:
        from fpdf import FPDF  # type: ignore
    except ImportError:
        return None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 8, f"Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(4)

    for msg in messages:
        role = str(msg.get("role", "unknown")).upper()
        content = str(msg.get("content", ""))
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, role, ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, content.encode("latin-1", "replace").decode("latin-1"))
        pdf.ln(3)

    return bytes(pdf.output(dest="S"))


def export_chat_buttons(
    messages: list[dict[str, Any]],
    *,
    filename_stem: str = "legal_assistant_chat",
    key_prefix: str = "export_chat",
) -> None:
    """Render PDF / Markdown / TXT export buttons for a chat transcript.

    Args:
        messages: List of chat message dicts (``role`` + ``content``).
        filename_stem: Base filename (without extension) for downloads.
        key_prefix: Prefix for widget keys, to disambiguate multiple
            instances of this component on the same page.
    """
    if not messages:
        st.caption("No conversation to export yet.")
        return

    col1, col2, col3 = st.columns(3)

    pdf_bytes = _chat_to_pdf_bytes(messages)
    with col1:
        if pdf_bytes is not None:
            st.download_button(
                "📄 Export PDF",
                data=pdf_bytes,
                file_name=f"{filename_stem}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"{key_prefix}_pdf",
            )
        else:
            st.button(
                "📄 Export PDF",
                disabled=True,
                use_container_width=True,
                help="Install the 'fpdf2' package to enable PDF export.",
                key=f"{key_prefix}_pdf_disabled",
            )

    with col2:
        st.download_button(
            "📝 Export Markdown",
            data=_chat_to_markdown(messages),
            file_name=f"{filename_stem}.md",
            mime="text/markdown",
            use_container_width=True,
            key=f"{key_prefix}_md",
        )

    with col3:
        st.download_button(
            "📃 Export TXT",
            data=_chat_to_txt(messages),
            file_name=f"{filename_stem}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"{key_prefix}_txt",
        )


# =============================================================================
# Retrieved evidence export — CSV / JSON
# =============================================================================


def _evidence_to_csv(evidence: list[dict[str, Any]]) -> str:
    """Render a list of retrieved-evidence dicts as CSV text.

    Column set is derived from the union of keys across all rows, so it
    adapts to whatever shape the backend's ``/retrieve`` results
    actually have rather than assuming fixed fields.
    """
    if not evidence:
        return ""

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in evidence:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in evidence:
        # Flatten any nested dict/list values to a JSON string so the CSV
        # stays well-formed regardless of payload shape.
        flat_row = {
            key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
            for key, value in row.items()
        }
        writer.writerow(flat_row)
    return buffer.getvalue()


def export_evidence_buttons(
    evidence: list[dict[str, Any]],
    *,
    filename_stem: str = "retrieved_evidence",
    key_prefix: str = "export_evidence",
) -> None:
    """Render CSV / JSON export buttons for retrieved evidence/chunks.

    Args:
        evidence: List of retrieved chunk/result dicts (e.g. from
            ``api_client.retrieve(...)["data"]["results"]``).
        filename_stem: Base filename (without extension) for downloads.
        key_prefix: Prefix for widget keys.
    """
    if not evidence:
        st.caption("No retrieved evidence to export yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📊 Export CSV",
            data=_evidence_to_csv(evidence),
            file_name=f"{filename_stem}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{key_prefix}_csv",
        )

    with col2:
        st.download_button(
            "🧾 Export JSON",
            data=json.dumps(evidence, indent=2, ensure_ascii=False),
            file_name=f"{filename_stem}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{key_prefix}_json",
        )


# =============================================================================
# Statistics export — CSV
# =============================================================================


def export_statistics_button(
    statistics: dict[str, Any],
    *,
    filename_stem: str = "system_statistics",
    key: str = "export_statistics_csv",
) -> None:
    """Render a CSV export button for a flat statistics dict.

    Nested values (e.g. ``endpoint_counts``) are expanded into their own
    rows so the CSV stays readable rather than embedding raw JSON.

    Args:
        statistics: The normalized statistics dict (e.g. from
            ``api_client.statistics()["data"]``).
        filename_stem: Base filename (without extension).
        key: Widget key.
    """
    if not statistics:
        st.caption("No statistics to export yet.")
        return

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Metric", "Value"])

    for metric_key, value in statistics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                writer.writerow([f"{metric_key}.{sub_key}", sub_value])
        else:
            writer.writerow([metric_key, value])

    st.download_button(
        "📊 Export Statistics (CSV)",
        data=buffer.getvalue(),
        file_name=f"{filename_stem}.csv",
        mime="text/csv",
        use_container_width=True,
        key=key,
    )