"""Inject missing VertexAI symbols so ragas can import on modern langchain-community."""
import sys
from types import ModuleType


def apply() -> None:
    # Import the REAL packages first (do not shadow them)
    import langchain_community
    import langchain_community.chat_models as chat_models

    # --- chat_models.vertexai.ChatVertexAI ---
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
    except Exception:
        vertexai_mod = ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # noqa: N801
            pass

        vertexai_mod.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mod
        setattr(chat_models, "vertexai", vertexai_mod)

    # --- llms.VertexAI ---
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:
        llms_name = "langchain_community.llms"
        if llms_name not in sys.modules:
            # If llms package is missing entirely, create a tiny stub package
            llms_mod = ModuleType(llms_name)
            llms_mod.__path__ = []  # mark as package
            sys.modules[llms_name] = llms_mod
            setattr(langchain_community, "llms", llms_mod)
        else:
            llms_mod = sys.modules[llms_name]

        class VertexAI:  # noqa: N801
            pass

        setattr(llms_mod, "VertexAI", VertexAI)