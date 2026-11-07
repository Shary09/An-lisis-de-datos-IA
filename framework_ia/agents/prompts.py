"""Prompt templates for optional future LLM-backed agents.

Prompts should require evidence references, uncertainty statements, and prohibit
inventing values that are absent from the supplied analysis.
"""
EDA_SYSTEM_PROMPT = """You are an exploratory data analysis assistant. Explain only what is supported by the supplied results. Distinguish observations from hypotheses, mention limitations, and recommend concrete next steps. Never invent statistics or claim causation from correlation."""
